# coding=utf-8
# Copyright (c) 2023 Alibaba PAI team.
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


import os
import tensorflow.compat.v1 as tf

from model import base_model, FLAGS, GptMoePreTrainedModel, PretrainPreprocessor
from utils import BundleTFRecordReader, get_shape_list


class GptMoePretrain(base_model):
    def __init__(self, **kwargs):
        super(GptMoePretrain, self).__init__(**kwargs)

    def build_logits(self, features, mode=None):
        preprocessor = PretrainPreprocessor(
            app_model_name="pretrain_language_model",
            feature_type="pretrain_lm",
            **self.config.__dict__)

        self.model = GptMoePreTrainedModel(**self.config.__dict__)
        print (self.config)

        input_ids, input_mask = preprocessor(features)
        target_ids = tf.concat([input_ids[:, 1:], tf.zeros_like(input_ids)[:, :1]], axis=-1)
        logits = self.model([input_ids, input_mask],
                            output_features=False,
                            mode=mode,
                            task_types=self.task_types)
        labels = target_ids

        return logits, labels

    def build_loss(self, logits, labels):
        loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(labels, [-1]))
        loss = tf.reduce_mean(loss_batch)
        return loss

    def build_eval_metrics(self, logits, labels):
        lm_log_probs = tf.nn.log_softmax(logits, axis=-1)
        lm_log_probs = tf.reshape(lm_log_probs, [-1, lm_log_probs.shape[-1]])
 
        target_ids = tf.reshape(labels, [-1])
        target_mask = 1 - tf.cast(tf.math.equal(target_ids, 0), tf.float32)
        one_hot_labels = tf.one_hot(
            target_ids, depth=self.model.config.vocab_size, dtype=tf.float32)
 
        lm_example_loss = -tf.reduce_sum(lm_log_probs * one_hot_labels, axis=[-1])
        lm_predictions = tf.argmax(lm_log_probs, axis=-1, output_type=tf.int32)
 
        lm_example_loss = tf.reshape(lm_example_loss, [-1])
        lm_accuracy = tf.metrics.accuracy(
            labels=target_ids, predictions=lm_predictions, weights=target_mask)
 
        lm_mean_loss = tf.metrics.mean(
            values=lm_example_loss, weights=target_mask)
 
        metric_dict = {
            "eval_{}_lm_accuracy".format(task): lm_accuracy,
            "eval_{}_lm_loss".format(task): lm_mean_loss
        }
 
        return metric_dict


    def build_predictions(self, output):
        lm_logits, _ = output
        return {"lm_logits": logits}


def main(_):
    # TePDist needs to use this semantics.
    tf.compat.v1.enable_resource_variables()

    app = GptMoePretrain()

    # if FLAGS.mode == "train_and_evaluate":
    if FLAGS.mode == "train":
        global_batch_size = FLAGS.train_batch_size
        train_reader = BundleTFRecordReader(input_glob=app.train_input_fp,
                                            is_training=True,
                                            shuffle_buffer_size=1024,
                                            input_schema=app.input_schema,
                                            batch_size=global_batch_size,
                                            fake_input=FLAGS.fake_input)

    # eval_reader = BundleTFRecordReader(input_glob=app.eval_input_fp,
    #                                     is_training=False,
    #                                     shuffle_buffer_size=1024,
    #                                     input_schema=app.input_schema,
    #                                     batch_size=app.eval_batch_size)

    # app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)
    app.run_train(reader=train_reader)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
