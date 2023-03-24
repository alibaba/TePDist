# coding=utf-8
# Copyright (c) 2019 Alibaba PAI team.
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

import copy
import json
import os
import tensorflow.compat.v1 as tf

from .flags import FLAGS, tffloat

from optimizers import get_train_op

class Config(object):
    def __init__(self, mode, config_json):
        self.mode = mode

        if self.mode == 'train' or self.mode == "train_and_evaluate" \
                or self.mode == "train_and_evaluate_on_the_fly" or self.mode == "train_on_the_fly":

            # optimizer
            self.optimizer = FLAGS.optimizer if FLAGS.optimizer else \
                str(config_json["train_config"]['optimizer_config'].get('optimizer', "adam"))
            self.learning_rate = float(config_json['train_config']['optimizer_config'].get('learning_rate', 0.001))
            self.opt_clip_after_allreduce = FLAGS.opt_clip_after_allreduce if FLAGS.opt_clip_after_allreduce is not None else \
                str(config_json['train_config']['optimizer_config'].get('opt_clip_after_allreduce', "False"))
            self.communication_fp16 = FLAGS.communication_fp16 if FLAGS.communication_fp16 \
                else str(config_json['train_config']['optimizer_config'].get('communication_fp16', "False"))
            self.weight_decay_ratio = float(
                config_json['train_config']['optimizer_config'].get('weight_decay_ratio', 0))
            self.lr_decay = config_json['train_config']['optimizer_config'].get('lr_decay', "polynomial")
            self.warmup_ratio = float(config_json['train_config']['optimizer_config'].get('warmup_ratio', 0.1))
            self.clip_norm_value = float(config_json['train_config']['optimizer_config'].get('clip_norm_value', 1.0))
            self.gradient_clip = FLAGS.gradient_clip if FLAGS.gradient_clip is not None else \
                (config_json['train_config']['optimizer_config'].get('gradient_clip', "global"))
            self.num_freezed_layers = int(config_json['train_config']['optimizer_config'].get('num_freezed_layers', 0))

            # misc
            self.num_epochs = float(config_json['train_config'].get('num_epochs', 1))
            try:
                self.model_dir = str(config_json['train_config'].get('model_dir', None)) if FLAGS.ckpt_dir is None else FLAGS.ckpt_dir
            except:
                raise ValueError("input model dir")

            self.throttle_secs = int(config_json['train_config'].get('throttle_secs', 100))
            self.keep_checkpoint_max = int(config_json['train_config'].get('keep_checkpoint_max', 10))

            if 'save_steps' not in config_json['train_config']:
                self.save_steps = None
            else:
                self.save_steps = int(config_json['train_config']['save_steps']) \
                    if config_json['train_config']['save_steps'] else \
                    config_json['train_config']['save_steps']

            self.log_step_count_steps = int(config_json['train_config'].get('log_step_count_steps', 100))

            # model
            for key, val in config_json['model_config'].items():
                setattr(self, key, val)

            # data
            self.input_schema = str(config_json['preprocess_config'].get('input_schema', None))

            if self.mode == 'train_and_evaluate_on_the_fly' or self.mode == 'train_on_the_fly':
                self.sequence_length = int(config_json['preprocess_config']['sequence_length'])
                self.first_sequence = str(config_json['preprocess_config']['first_sequence'])
                self.second_sequence = str(config_json['preprocess_config'].get('second_sequence', None))
                self.label_name = str(config_json['preprocess_config']['label_name'])
                self.label_enumerate_values = config_json['preprocess_config'].get('label_enumerate_values', None)
                self.append_feature_columns = config_json['preprocess_config'].get('append_feature_columns', None)

            if self.mode == 'train_and_evaluate' or self.mode == 'train_and_evaluate_on_the_fly':
                self.eval_batch_size = int(config_json['evaluate_config']['eval_batch_size'])

                if 'num_eval_steps' not in config_json['evaluate_config']:
                    self.num_eval_steps = None
                else:
                    self.num_eval_steps = int(config_json['evaluate_config']['num_eval_steps']) \
                        if config_json['evaluate_config']['num_eval_steps'] else \
                        config_json['evaluate_config']['num_eval_steps']

                self.eval_input_fp = str(config_json['evaluate_config']['eval_input_fp'])

            self.train_input_fp = str(config_json['train_config']['train_input_fp'])
            self.train_batch_size = FLAGS.train_batch_size if FLAGS.train_batch_size \
                else int(config_json['train_config']['train_batch_size'])

        elif self.mode == "evaluate" or self.mode == "evaluate_on_the_fly":
            self.eval_ckpt_path = config_json['evaluate_config']['eval_checkpoint_path']

            self.input_schema = config_json['preprocess_config']['input_schema']
            if self.mode == "evaluate_on_the_fly":
                self.sequence_length = int(config_json['preprocess_config']['sequence_length'])
                self.first_sequence = str(config_json['preprocess_config']['first_sequence'])
                self.second_sequence = str(config_json['preprocess_config']['second_sequence'])
                self.label_name = str(config_json['preprocess_config']['label_name'])
                self.label_enumerate_values = config_json['preprocess_config'].get('label_enumerate_values', None)

            for key, val in config_json['model_config'].items():
                setattr(self, key, val)

            self.model_dir = str(config_json['train_config'].get('model_dir', None))
            self.eval_batch_size = config_json['evaluate_config']['eval_batch_size']
            self.num_eval_steps = config_json['evaluate_config'].get('num_eval_steps', None)
            self.eval_input_fp = config_json['evaluate_config']['eval_input_fp']

        elif self.mode == 'predict' or self.mode == 'predict_on_the_fly':
            self.num_gpu_parallel = int(config_json["predict_config"].get("distribution_config", {}).get(
                "num_gpu_parallel", 2))
            self.predict_checkpoint_path = config_json['predict_config'].get('predict_checkpoint_path', None)
            self.input_schema = config_json['preprocess_config']['input_schema']
            self.label_name = config_json['preprocess_config'].get('label_name', None)
            self.label_enumerate_values = config_json['preprocess_config'].get('label_enumerate_values', None)
            self.append_feature_columns = config_json['preprocess_config'].get('append_feature_columns', None)
            self.model_dir = config_json.get('train_config', {}).get('model_dir', None)
            self.output_schema = config_json['preprocess_config'].get('output_schema', None)

            if self.mode == 'predict_on_the_fly':
                self.first_sequence = config_json['preprocess_config']['first_sequence']
                self.second_sequence = config_json['preprocess_config'].get('second_sequence', None)
                self.sequence_length = config_json['preprocess_config']['sequence_length']
                self.max_predictions_per_seq = config_json['preprocess_config'].get('max_predictions_per_seq', None)

            self.predict_batch_size = config_json['predict_config']['predict_batch_size']

            if config_json['preprocess_config'].get('output_schema', None) == "bert_finetune":
                self.output_schema = "input_ids,input_mask,segment_ids,label_id"

            elif config_json['preprocess_config'].get('output_schema', None) == "bert_pretrain":
                self.output_schema = "input_ids,input_mask,segment_ids,masked_lm_positions,masked_lm_ids,masked_lm_weights"

            elif config_json['preprocess_config'].get('output_schema', None) == "bert_predict":
                self.output_schema = "input_ids,input_mask,segment_ids"
            else:
                self.output_schema = config_json['preprocess_config'].get('output_schema', None)

            self.model_config = config_json['model_config']
            for key, val in config_json['model_config'].items():
                setattr(self, key, val)

            self.predict_input_fp = config_json['predict_config']['predict_input_fp']
            self.predict_output_fp = config_json['predict_config'].get('predict_output_fp', None)

        elif self.mode == 'export':
            self.checkpoint_path = config_json['export_config']['checkpoint_path']
            for key, val in config_json['model_config'].items():
                setattr(self, key, val)
            self.export_dir_base = config_json['export_config']['export_dir_base']
            self.checkpoint_path = config_json['export_config']['checkpoint_path']
            self.receiver_tensors_schema = config_json['export_config']['receiver_tensors_schema']
            self.input_tensors_schema = config_json['export_config']['input_tensors_schema']

        elif self.mode == 'preprocess':
            self.input_schema = config_json['preprocess_config']['input_schema']
            self.first_sequence = config_json['preprocess_config']['first_sequence']
            self.second_sequence = config_json['preprocess_config'].get('second_sequence', None)
            self.label_name = config_json['preprocess_config'].get('label_name', None)
            self.label_enumerate_values = config_json['preprocess_config'].get('label_enumerate_values', None)
            self.sequence_length = config_json['preprocess_config']['sequence_length']
            self.max_predictions_per_seq = config_json['preprocess_config'].get('max_predictions_per_seq', None)

            if config_json['preprocess_config']['output_schema'] == "bert_finetune":
                self.output_schema = "input_ids,input_mask,segment_ids,label_id"

            elif config_json['preprocess_config']['output_schema'] == "bert_pretrain":
                self.output_schema = "input_ids,input_mask,segment_ids,masked_lm_positions,masked_lm_ids,masked_lm_weights"

            elif config_json['preprocess_config']['output_schema'] == "bert_predict":
                self.output_schema = "input_ids,input_mask,segment_ids"
            else:
                self.output_schema = config_json['preprocess_config']['output_schema']

            self.preprocess_input_fp = config_json['preprocess_config']['preprocess_input_fp']
            self.preprocess_output_fp = config_json['preprocess_config']['preprocess_output_fp']
            self.preprocess_batch_size = config_json['preprocess_config']['preprocess_batch_size']
            self.tokenizer_name_or_path = config_json['preprocess_config']['tokenizer_name_or_path']

    def __str__(self):
        return json.dumps(self.__dict__, sort_keys=False, indent=4)


class base_model(object):
    def __init__(self, **kwargs):
        user_defined_config = kwargs.get("user_defined_config", None)
        if user_defined_config is None:
            assert FLAGS.mode is not None
            with tf.gfile.Open(FLAGS.config, "r") as f:
                tf.logging.info("config file is {}".format(FLAGS.config))
                config_json = json.load(f)

            if "predict" in FLAGS.mode:
                if config_json['predict_config'].get('predict_checkpoint_path', None) is not None:
                    model_ckpt = config_json['predict_config']['predict_checkpoint_path'].split("/")[-1]
                    config_fp = config_json['predict_config']['predict_checkpoint_path'].replace(model_ckpt,
                                                                                                 "train_config.json")
                    if tf.gfile.Exists(config_fp):
                        with tf.gfile.Open(config_fp, "r") as f:
                            saved_config = json.load(f)
                            model_config = saved_config.get("model_config", None)
                            config_json["model_config"] = model_config

            self.config = Config(mode=FLAGS.mode, config_json=config_json)

            if "train" in FLAGS.mode:
                assert self.config.model_dir is not None
                if not tf.gfile.Exists(self.config.model_dir):
                    tf.gfile.MakeDirs(self.config.model_dir)

                if not tf.gfile.Exists(self.config.model_dir + "/train_config.json"):
                    with tf.gfile.GFile(self.config.model_dir + "/train_config.json", mode='w') as f:
                        json.dump(config_json, f)

        else:
            self.config = user_defined_config

        for key, val in self.config.__dict__.items():
            setattr(self, key, val)

        self.num_train_examples = 99968*1000
        self.num_predict_examples = 0

        if self.config.mode == 'train' or self.config.mode == "train_and_evaluate" or \
                self.config.mode == "train_and_evaluate_on_the_fly" or self.config.mode == "train_on_the_fly":

            tf.logging.info("***********Running in {} mode***********".format(self.config.mode))
            self.strategy = None
            global_batch_size = self.config.train_batch_size
            tf.logging.info("***********TePDist distribution strategy***********")

            # if save steps is None, save per epoch
            if self.config.save_steps is None:
                self.save_steps = int(self.num_train_examples / global_batch_size)
            else:
                self.save_steps = self.config.save_steps

            self.train_steps = FLAGS.stop_at_step \
                if FLAGS.stop_at_step else \
                int(self.num_train_examples *
                    self.config.num_epochs / global_batch_size) + 1

            self.throttle_secs = self.config.throttle_secs
            self.model_dir = self.config.model_dir
            tf.logging.info("model_dir: {}".format(self.config.model_dir))
            tf.logging.info("learning rate: {}".format(self.config.learning_rate))
            tf.logging.info("train batch size: {}".format(self.config.train_batch_size))
            tf.logging.info("global batch size: {}".format(global_batch_size))
            tf.logging.info("num train examples per epoch: {}".format(self.num_train_examples))
            tf.logging.info("num epochs: {}".format(self.config.num_epochs))
            tf.logging.info("train steps: {}".format(self.train_steps))
            tf.logging.info("save steps: {}".format(self.save_steps))
            tf.logging.info("throttle secs: {}".format(self.throttle_secs))
            tf.logging.info("keep checkpoint max: {}".format(self.config.keep_checkpoint_max))
            tf.logging.info("warmup ratio: {}".format(self.config.warmup_ratio))
            tf.logging.info("gradient clip: {}".format(self.config.gradient_clip))
            tf.logging.info("clip norm value: {}".format(self.config.clip_norm_value))
            tf.logging.info("log step count steps: {}".format(self.config.log_step_count_steps))


            self.estimator = tf.estimator.Estimator(
                model_fn=self._build_model_fn(),
                model_dir=self.config.model_dir,
                config=self._get_run_train_config(config=self.config))

            if self.config.mode == 'train_and_evaluate' or self.config.mode == 'train_and_evaluate_on_the_fly':
                self.num_eval_steps = self.config.num_eval_steps
                tf.logging.info("num eval steps: {}".format(self.num_eval_steps))

        elif self.config.mode == 'evaluate' or self.config.mode == 'evaluate_on_the_fly':
            self.num_eval_steps = self.config.num_eval_steps
            tf.logging.info("num eval steps: {}".format(self.num_eval_steps))
            tf.logging.info("***********Running in {} mode***********".format(self.config.mode))
            self.estimator = tf.estimator.Estimator(
                model_fn=self._build_model_fn(),
                config=self._get_run_predict_config())

        elif self.config.mode == 'predict' or self.config.mode == 'predict_on_the_fly':
            tf.logging.info("***********Running in {} mode***********".format(self.config.mode))
            self.estimator = tf.estimator.Estimator(
                model_fn=self._build_model_fn(),
                config=self._get_run_predict_config())

        elif self.config.mode == 'export':
            tf.logging.info("***********Running in {} mode***********".format(self.config.mode))
            self.estimator = tf.estimator.Estimator(
                model_fn=self._build_model_fn(),
                config=self._get_run_predict_config())

        elif self.config.mode == 'preprocess':
            tf.logging.info("***********Running in {} mode***********".format(self.config.mode))
            self.estimator = tf.estimator.Estimator(
                model_fn=self._build_model_fn(),
                config=tf.estimator.RunConfig())

            self.first_sequence = self.config.first_sequence
            self.second_sequence = self.config.second_sequence
            self.label_enumerate_values = self.config.label_enumerate_values
            self.label_name = self.config.label_name


    def get_export_features(self):
        export_features = {}

        for feat in self.config.input_tensors_schema.split(","):
            feat_name = feat.split(":")[0]
            feat_type = feat.split(":")[1]
            seq_len = int(feat.split(":")[2])
            feat = {}
            feat['name'] = feat_name
            feat['type'] = feat_type
            if feat_type == "int":
                dtype = tf.int32
            elif feat_type == "float":
                dtype = tffloat
            if seq_len == 1:
                ph = tf.placeholder(dtype=dtype, shape=[None], name=feat_name)
            else:
                ph = tf.placeholder(dtype=dtype, shape=[None, None], name=feat_name)

            export_features[feat_name] = ph

        receiver_tensors = {}
        feat_names = []
        for feat in self.config.receiver_tensors_schema.split(","):
            feat_names.append(feat.split(":")[0])
        for feat_name in feat_names:
            receiver_tensors[feat_name] = export_features[feat_name]
        return export_features, receiver_tensors

    def build_logits(self, features, mode):
        """ Given features, this method take care of building graph for train/eval/predict

        Args:

            features : either raw text features or numerical features such as input_ids, input_mask ...
            mode : tf.estimator.ModeKeys.TRAIN | tf.estimator.ModeKeys.EVAL | tf.estimator.ModeKeys.PREDICT

        Returns:

            logits, labels

        Examples::

            def build_logits(self, features, mode=None):
                preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path)
                model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

                dense = layers.Dense(self.num_labels,
                                     kernel_initializer=layers.get_initializer(0.02),
                                     name='dense')

                input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
                outputs = model([input_ids, input_mask, segment_ids], mode=mode)
                pooled_output = outputs[1]

                logits = dense(pooled_output)
                return logits, label_ids

        """
        raise NotImplementedError("must be implemented in descendants")

    def build_loss(self, logits, labels):
        """Build loss function

        Args:

            logits : logits returned from build_logits
            labels : labels returned from build_logits

        Returns:

            loss

        Examples::

            def build_loss(self, logits, labels):
                return softmax_cross_entropy(labels, depth=self.config.num_labels, logits=logits)

        """

        raise NotImplementedError("must be implemented in descendants")

    def build_eval_metrics(self, logits, labels):
        """Build evaluation metrics

        Args:

            logits : logits returned from build_logits
            labels : labels returned from build_logits

        Returns:

            metric_dict

        Examples::

            def build_eval_metrics(self, logits, labels):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                info_dict = {
                    "predictions": predictions,
                    "labels": labels,
                }
                evaluator = PyEvaluator()
                labels = [i for i in range(self.num_labels)]
                metric_dict = evaluator.get_metric_ops(info_dict, labels)
                ret_metrics = evaluator.evaluate(labels)
                tf.summary.scalar("eval accuracy", ret_metrics['py_accuracy'])
                tf.summary.scalar("eval F1 micro score", ret_metrics['py_micro_f1'])
                tf.summary.scalar("eval F1 macro score", ret_metrics['py_macro_f1'])
                return metric_dict

        """

        raise NotImplementedError("must be implemented in descendants")

    def build_predictions(self, logits):
        """Build predictions

        Args:

            logits : logits returned from build_logits


        Returns:

            predictions

        Examples::

            def build_predictions(self, output):
                logits, _ = output
                predictions = dict()
                predictions["predictions"] = tf.argmax(logits, axis=-1, output_type=tf.int32)
                return predictions

        """
        raise NotImplementedError("must be implemented in descendants")

    def _get_run_train_config(self, config):
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=64,
            inter_op_parallelism_threads=64,
            gpu_options=tf.GPUOptions(allow_growth=True,
                                      force_gpu_compatible=True,
                                      per_process_gpu_memory_fraction=1.0))

        # To avoid _FusedMatMul which cannot be lowered by XLA.
        from tensorflow.core.protobuf import rewriter_config_pb2
        off = rewriter_config_pb2.RewriterConfig.OFF
        session_config.graph_options.rewrite_options.remapping = off

        session_config.graph_options.rewrite_options.memory_optimization = off

        # Disable grappler optimizers timeout check.
        # This is necessary for very large models with big graph, where grappler
        # optimizers will take longer time.
        # For example, M6 with 8 encoder layers and 512 experts (30B) will trigger
        # grappler timeout check.
        session_config.graph_options.rewrite_options.meta_optimizer_timeout_ms = -1

        if FLAGS.dist_hlo_baseline:
          # TePDist distributed variable initialization.
          session_config.graph_options.rewrite_options.init_from_remote = off

        if FLAGS.enable_amp:
          session_config.graph_options.rewrite_options.auto_mixed_precision = 1
          # Move ArgMax from AMP clearlist to backlist as ArgMax of fp16 type
          # has no XLAKernel support and cannot be clustered to Whole graph.
          os.environ["TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_CLEARLIST_REMOVE"] = "ArgMax"
          os.environ["TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_BLACKLIST_ADD"] = "ArgMax"
        # Constant folding grappler pass
        #session_config.graph_options.rewrite_options.constant_folding = off
        run_config = tf.estimator.RunConfig(session_config=session_config,
                                            model_dir=config.model_dir,
                                            tf_random_seed=123123,
                                            train_distribute=self.strategy,
                                            log_step_count_steps=1,
                                            # log_step_count_steps=100,
                                            # save_checkpoints_steps=self.save_steps,
                                            save_checkpoints_steps=None,
                                            save_checkpoints_secs=None,
                                            keep_checkpoint_max=config.keep_checkpoint_max
                                            )
        return run_config

    def _get_run_predict_config(self):

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=64,
            inter_op_parallelism_threads=64,
            gpu_options=tf.GPUOptions(allow_growth=True,
                                      force_gpu_compatible=True,
                                      per_process_gpu_memory_fraction=1.0))

        run_config = tf.estimator.RunConfig(session_config=session_config)
        return run_config

    def _build_model_fn(self):
        def model_fn(features, labels, mode, params):
            if mode == tf.estimator.ModeKeys.TRAIN:
                logits, labels = self.build_logits(features, mode=mode)
                total_loss = self.build_loss(logits, labels)
                train_op, learning_rate = get_train_op(learning_rate=self.config.learning_rate,
                                        weight_decay_ratio=self.config.weight_decay_ratio,
                                        loss=total_loss,
                                        lr_decay=self.config.lr_decay,
                                        warmup_ratio=self.config.warmup_ratio,
                                        optimizer_name=self.optimizer,
                                        tvars=self.tvars if hasattr(self, "tvars") else None,
                                        train_steps=self.train_steps,
                                        clip_norm=self.config.gradient_clip,
                                        clip_norm_value=self.config.clip_norm_value,
                                        num_freezed_layers=self.config.num_freezed_layers,
                                        enable_amp=FLAGS.enable_amp,
                                        enable_ga=FLAGS.enable_ga
                                        )

                # summary_hook = tf.train.SummarySaverHook(save_steps=100, summary_op=tf.summary.merge_all())

                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=total_loss, train_op=train_op, 
                    training_hooks=[])

            elif mode == tf.estimator.ModeKeys.EVAL:
                logits, labels = self.build_logits(features, mode=mode)
                eval_loss = self.build_loss(logits, labels)
                #tf.summary.scalar("eval_loss", eval_loss)
                metrics = self.build_eval_metrics(logits, labels)
                #summary_hook = tf.train.SummarySaverHook(save_steps=100,
                #                                         summary_op=tf.summary.merge_all())
                return tf.estimator.EstimatorSpec(mode, loss=eval_loss,
                                                  eval_metric_ops=metrics,
                                                  evaluation_hooks=[])#summary_hook])

            elif mode == tf.estimator.ModeKeys.PREDICT:
                if self.config.mode == 'predict' or self.config.mode == 'export':
                    output = self.build_logits(features, mode=mode)
                    predictions = self.build_predictions(output)

                elif self.config.mode == 'predict_on_the_fly':
                    output = self.build_logits(features, mode=mode)
                    predictions = self.build_predictions(output)

                elif self.config.mode == 'preprocess':
                    output = self.build_logits(features, mode=mode)
                    predictions = self.build_predictions(output)
                else:
                    predictions = features

                output = {'serving_default': tf.estimator.export.PredictOutput(predictions)}
                predictions.update(features)

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    export_outputs=output)

        return model_fn

    def run_train_and_evaluate(self, train_reader, eval_reader):
        train_spec = tf.estimator.TrainSpec(input_fn=train_reader.get_input_fn(),
                                            max_steps=self.train_steps)

        eval_spec = tf.estimator.EvalSpec(input_fn=eval_reader.get_input_fn(),
                                          steps=self.num_eval_steps,
                                          throttle_secs=self.throttle_secs)

        tf.logging.info("*********Calling tf.estimator.train_and_evaluate *********")
        tf.estimator.train_and_evaluate(self.estimator,
                                        train_spec=train_spec,
                                        eval_spec=eval_spec)

    def run_train(self, reader):
        self.estimator.train(input_fn=reader.get_input_fn(),
                             max_steps=self.train_steps)
                             #log_every_step=FLAGS.log_every_step)

    def run_evaluate(self, reader, checkpoint_path=None):
        return self.estimator.evaluate(input_fn=reader.get_input_fn(),
                                       steps=self.num_eval_steps,
                                       checkpoint_path=checkpoint_path)

    def run_predict(self, reader, writer=None, checkpoint_path=None, yield_single_examples=False):

        if writer is None:
            return self.estimator.predict(
                input_fn=reader.get_input_fn(),
                yield_single_examples=yield_single_examples,
                checkpoint_path=checkpoint_path)

        for batch_idx, outputs in enumerate(self.estimator.predict(input_fn=reader.get_input_fn(),
                                                                   yield_single_examples=yield_single_examples,
                                                                   checkpoint_path=checkpoint_path)):

            if batch_idx % 1 == 0:
                tf.logging.info("Processing %d batches" % (batch_idx))
            writer.process(outputs)

        writer.close()

    def run_preprocess(self, reader, writer):
        for batch_idx, outputs in enumerate(self.estimator.predict(input_fn=reader.get_input_fn(),
                                                                   yield_single_examples=False,
                                                                   checkpoint_path=None)):
            if batch_idx % 100 == 0:
                tf.logging.info("Processing %d batches" % (batch_idx))
            writer.process(outputs)

        writer.close()

    def export_model(self):

        export_dir_base = self.config.export_dir_base
        checkpoint_path = self.config.checkpoint_path

        def serving_input_receiver_fn():
            export_features, receiver_tensors = self.get_export_features()
            return tf.estimator.export.ServingInputReceiver(
                features=export_features, receiver_tensors=receiver_tensors, receiver_tensors_alternatives={})

        return self.estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn,
                                                checkpoint_path=checkpoint_path)


