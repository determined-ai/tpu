# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a EfficientNets on ImageNet on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2  # used for summaries only.

import imagenet_input
import model_builder_factory
import utils

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.estimator import estimator

from determined.estimator import EstimatorTrial, EstimatorTrialContext


FAKE_DATA_DIR="gs://cloud-tpu-test-datasets/fake_imagenet"


def make_model_fn(context):
    def model_fn(features, labels, mode, params=None):
        """The model_fn to be used with TPUEstimator.

        Args:
            features: `Tensor` of batched images.
            labels: `Tensor` of one hot labels for the data samples
            mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`

        Returns:
        A `TPUEstimatorSpec` for the model
        """
        if isinstance(features, dict):
            features = features["feature"]

        # In most cases, the default data format NCHW instead of NHWC should be
        # used for a significant performance boost on GPU. NHWC should be used
        # only if the network needs to be run on CPU since the pooling operations
        # are only supported on NHWC. TPU uses XLA compiler to figure out best layout.
        if context.get_hparam("data_format") == "channels_first":
            assert not context.get_hparam("transpose_input")  # channels_first only for GPU
            features = tf.transpose(features, [0, 3, 1, 2])
            stats_shape = [3, 1, 1]
        else:
            stats_shape = [1, 1, 3]

        #if context.get_hparam("transpose_input") and mode != tf.estimator.ModeKeys.PREDICT:
        #    features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

        is_training = mode == tf.estimator.ModeKeys.TRAIN
        has_moving_average_decay = context.get_hparam("moving_average_decay") > 0
        # This is essential, if using a keras-derived model.
        tf.keras.backend.set_learning_phase(is_training)
        logging.info("Using open-source implementation.")
        override_params = {}
        #if context.get_hparam("batch_norm_momentum") is not None:
        #    override_params["batch_norm_momentum"] = context.get_hparam("batch_norm_momentum")
        #if context.get_hparam("batch_norm_epsilon") is not None:
        #    override_params["batch_norm_epsilon"] = context.get_hparam("batch_norm_epsilon")
       # if context.get_hparam("dropout_rate") is not None:
       #     override_params["dropout_rate"] = context.get_hparam("dropout_rate")
       # if context.get_hparam("survival_prob") is not None:
       #     override_params["survival_prob"] = context.get_hparam("survival_prob")
       # if context.get_hparam("data_format"):
       #     override_params["data_format"] = context.get_hparam("data_format")
       # if context.get_hparam("num_label_classes"):
       #     override_params["num_classes"] = context.get_hparam("num_label_classes")
       # if context.get_hparam("depth_coefficient"):
       #     override_params["depth_coefficient"] = context.get_hparam("depth_coefficient")
       # if context.get_hparam("width_coefficient"):
       #     override_params["width_coefficient"] = context.get_hparam("width_coefficient")

        def normalize_features(features, mean_rgb, stddev_rgb):
            """Normalize the image given the means and stddevs."""
            features -= tf.constant(mean_rgb, shape=stats_shape, dtype=features.dtype)
            features /= tf.constant(stddev_rgb, shape=stats_shape, dtype=features.dtype)
            return features

        def build_model():
            """Build model using the model_name given through the command line."""
            model_builder = model_builder_factory.get_model_builder(
                context.get_hparam("model_name"),
            )
            normalized_features = normalize_features(
                features, model_builder.MEAN_RGB, model_builder.STDDEV_RGB
            )
            logits, _ = model_builder.build_model(
                normalized_features,
                model_name=context.get_hparam("model_name"),
                training=is_training,
                override_params=override_params,
                #model_dir=context.get_hparam("model_dir"),
            )
            return logits

        logits = build_model()

        # Calculate loss, which includes softmax cross entropy and L2 regularization.
        cross_entropy = tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=labels, label_smoothing=context.get_hparam("label_smoothing")
        )

        # Add weight decay to the loss for non-batch-normalization variables.
        loss = cross_entropy + context.get_hparam("weight_decay") * tf.add_n(
            [
                tf.nn.l2_loss(v)
                for v in tf.trainable_variables()
                if "batch_normalization" not in v.name
            ]
        )

        global_step = tf.train.get_global_step()
        if has_moving_average_decay:
            ema = tf.train.ExponentialMovingAverage(
                decay=context.get_hparam("moving_average_decay"), num_updates=global_step
            )
            ema_vars = utils.get_ema_vars()

        restore_vars_dict = None
        train_op = None
        if is_training:
            # Compute the current epoch and associated learning rate from global_step.
            current_epoch = tf.cast(global_step, tf.float32) / context.get_hparam("steps_per_epoch")

            scaled_lr = context.get_hparam("base_learning_rate") * (context.get_hparam("train_batch_size") / 256.0)
            logging.info("base_learning_rate = %f", context.get_hparam("base_learning_rate"))
            learning_rate = utils.build_learning_rate(
                scaled_lr, global_step, context.get_hparam("steps_per_epoch"),
            )
            optimizer = utils.build_optimizer(learning_rate)

            # Batch normalization requires UPDATE_OPS to be added as a dependency to
            # the train operation.
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step)

            if has_moving_average_decay:
                with tf.control_dependencies([train_op]):
                    train_op = ema.apply(ema_vars)

        if has_moving_average_decay:
            # Load moving average variables for eval.
            restore_vars_dict = ema.variables_to_restore(ema_vars)

        eval_metrics = None
        if mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(labels, logits):
                """Evaluation metric function. Evaluates accuracy.

                This function is executed on the CPU and should not directly reference
                any Tensors in the rest of the `model_fn`. To pass Tensors from the model
                to the `metric_fn`, provide as part of the `eval_metrics`. See
                https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
                for more information.

                Arguments should match the list of `Tensor` objects passed as the second
                element in the tuple passed to `eval_metrics`.

                Args:
                    labels: `Tensor` with shape `[batch, num_classes]`.
                    logits: `Tensor` with shape `[batch, num_classes]`.

                Returns:
                    A dict of the metrics to return from evaluation.
                """
                labels = tf.argmax(labels, axis=1)
                predictions = tf.argmax(logits, axis=1)
                top_1_accuracy = tf.metrics.accuracy(labels, predictions)
                in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
                top_5_accuracy = tf.metrics.mean(in_top_5)

                return {
                    "top_1_accuracy": top_1_accuracy,
                    "top_5_accuracy": top_5_accuracy,
                }

            eval_metrics = metric_fn(labels, logits)

        num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        logging.info("number of trainable parameters: %d", num_params)


        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metrics,
        )
    return model_fn


def build_imagenet_input(context, is_training):
    input_image_size = model_builder_factory.get_model_input_size(context.get_hparam("model_name"))
    include_background_label = (context.get_hparam("num_label_classes") == 1001)
    """Generate ImageNetInput for training and eval."""
    data_dir = context.get_data_config().get("data_dir")
    logging.info("Using dataset: %s", data_dir)

    return imagenet_input.ImageNetInput(
        is_training=is_training,
        data_dir=data_dir,
        transpose_input=False,#context.get_hparam("transpose_input"),
        cache=False,#context.get_hparam("use_cache") and is_training,
        image_size=input_image_size,
        num_parallel_calls=context.get_hparam("num_parallel_calls"),
        num_label_classes=context.get_hparam("num_label_classes"),
        include_background_label=include_background_label,
        #augment_name=context.get_hparam("augment_name"),
        mixup_alpha=context.get_hparam("mixup_alpha"),
        randaug_num_layers=context.get_hparam("randaug_num_layers"),
        randaug_magnitude=context.get_hparam("randaug_magnitude"),
        resize_method=None,
        use_bfloat16=False,
        context=context,
    )


class EfficientNetEstimator(EstimatorTrial):
    def __init__(self, context):
        self.context = context

    def build_estimator(self) -> tf.estimator.Estimator:
        config = tf.estimator.RunConfig(save_summary_steps=1000)

        estimator = tf.estimator.Estimator(
            model_fn=make_model_fn(self.context), config=config, params=None
        )
        return estimator

    def build_train_spec(self) -> tf.estimator.TrainSpec:
        data = build_imagenet_input(self.context, True)
        return tf.estimator.TrainSpec(data.input_fn)

    def build_validation_spec(self) -> tf.estimator.EvalSpec:
        data = build_imagenet_input(self.context, False)
        return tf.estimator.EvalSpec(data.input_fn)
