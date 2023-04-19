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

import random
import numpy as np
import os
import tensorflow.compat.v1 as tf

SEED = 123123
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.reset_default_graph()
tf.set_random_seed(SEED)

flags = tf.app.flags
flags.DEFINE_string("config", default=None, help='')
flags.DEFINE_string("outputs", default=None, help='')
flags.DEFINE_string('job_name', 'worker', 'job_name')
flags.DEFINE_string("mode", default=None, help="Which mode")
flags.DEFINE_string("checkpoint_dir", default=None, help='')
flags.DEFINE_bool("use_fp16", default=False, help="Whether to use fp16")
flags.DEFINE_bool("fake_input", default=False, help="Whether to use fake input")
flags.DEFINE_integer("stop_at_step", 0, "")
flags.DEFINE_integer("train_batch_size", default=128, help='train_batch_size')
flags.DEFINE_integer("eval_batch_size", default=128, help='eval_batch_size')
flags.DEFINE_string("ckpt_dir", None, "")
flags.DEFINE_string("communication_fp16", None, "")
flags.DEFINE_integer("log_every_step", 100, "")
flags.DEFINE_integer("enable_amp", 0, "")
flags.DEFINE_integer("enable_radical_amp", 0, "")
flags.DEFINE_integer("enable_ga", 0, "")
flags.DEFINE_integer("enable_zero", 0, "")
flags.DEFINE_string("gradient_clip", None, "")
flags.DEFINE_integer("dist_hlo_baseline", 0, "Only use for baseline test on 1 gpu.")
flags.DEFINE_string("opt_clip_after_allreduce", None, "")
flags.DEFINE_string("optimizer", None, "")
flags.DEFINE_integer("profiler_steps", 0, "")
flags.DEFINE_integer("memory_profile_steps", 0, "")
FLAGS = flags.FLAGS

tffloat = tf.float16 if FLAGS.use_fp16 else tf.float32
if FLAGS.use_fp16:
  tf.keras.backend.set_floatx('float16')


