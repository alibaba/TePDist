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

import os
from glob import glob
import tensorflow.compat.v1 as tf
import numpy as np
import pickle
from collections import OrderedDict
squence_length = 512

class TFRecordReader(object):
    """ Read tfrecords

    Args:

        input_glob : input file fp
        batch_size : input batch size
        is_training : True or False
    """

    def __init__(self,
                 input_glob,
                 batch_size,
                 is_training,
                 job_name='DISTTFRecordReader',
                 **kwargs):

        self.num_train_examples = 0
        self.num_eval_examples = 0

        self.is_training = is_training
        self.num_parallel_batches = kwargs.pop("num_parallel_batches", 1)
        self.shuffle_buffer_size = kwargs.pop("shuffle_buffer_size", None)
        self.prefetch_buffer_size = kwargs.pop("prefetch_buffer_size", 1)
        self.input_schema = kwargs.pop("input_schema", None)
        # for all mode, generate tf.Tensor placeholders
        # all mode need a input_schema, column_name:type:length
        self.input_tensors = OrderedDict()
        self.input_tensor_names = []
        for schema in self.input_schema.split(","):
            name = schema.split(":")[0]
            self.input_tensor_names.append(name)
            type = schema.split(":")[1]
            seq_len = int(schema.split(":")[2])
            if type == "int":
                tensor_type = tf.int64
                default_value = 0
            elif type == "float":
                tensor_type = tf.float32
                default_value = 0.0
            elif type == "str":
                tensor_type = tf.string
                default_value = ''
            elif type == "base64":
                tensor_type = "base64"
                default_value = "base64"
            else:
                raise ValueError("unsupported feature type")

            self.input_tensors[name] = tf.io.FixedLenFeature([seq_len], tensor_type, default_value)

        self.batch_size = batch_size

        tf.logging.info("num_parallel_batches {}".format(self.num_parallel_batches))
        tf.logging.info("shuffle_buffer_size {}".format(self.shuffle_buffer_size))
        tf.logging.info("prefetch_buffer_size {}".format(self.prefetch_buffer_size))
        tf.logging.info("batch_size {}".format(self.batch_size))
        tf.logging.info("input_schema {}".format(self.input_schema))

        self.input_glob = input_glob
        self.num_train_examples = 99968*1000

    def _get_data_pipeline(self, dataset, _decode_fn):
        if self.is_training:
            if self.shuffle_buffer_size is None:
                tf.logging.info("Random shuffle on the whole {} training examples".format(self.num_train_examples))
                self.shuffle_buffer_size = self.num_train_examples
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=123123)
        else:
            dataset = dataset.repeat(1)

        return self._map_batch_prefetch(dataset, _decode_fn)

    def _map_batch_prefetch(self, dataset, decode_fn):
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                lambda *record: decode_fn(*record),
                batch_size=self.batch_size,
                num_parallel_batches=self.num_parallel_batches,
                drop_remainder=False))
        dataset = dataset.prefetch(self.prefetch_buffer_size)
        return dataset

    def get_input_fn(self):
        def input_fn():
            dataset = tf.data.TFRecordDataset(self.input_glob)
            return self._get_data_pipeline(dataset, self._decode_tfrecord)

        return input_fn

    def _decode_tfrecord(self, record):
        name_to_features = {}
        for name, feature in self.input_tensors.items():
            name_to_features[name] = tf.io.FixedLenFeature(feature.shape, feature.dtype, None)
        example = tf.parse_single_example(record, name_to_features)
        return example


class BundleTFRecordReader(TFRecordReader):
    def __init__(self, input_glob, batch_size, is_training=False, fake_input=False, model_type=None,**kwargs):
        super(BundleTFRecordReader, self).__init__(input_glob, batch_size, is_training, **kwargs)

        # for diffusion
        self.img_size = 224
        self.timesteps = 1000
        scale = 1000 / self.timesteps
        self.beta_start = scale * 0.0001
        self.beta_end = scale * 0.02

        self.fake_input = fake_input
        self.model_type = model_type
        self.input_fps = []
        with tf.gfile.Open(input_glob, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '' or line.isdigit():
                    continue
                self.input_fps.append(line)
        files = self.input_fps
        files.sort()
        self.files = files
        print("======self.input_fps:", self.input_fps, " self.files:", self.files, " num_files:", len(self.files))

    # For diffusion image process
    def adjust_dynamic_range(self, images, range_in, range_out, out_dtype):
        scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
        bias = range_out[0] - range_in[0] * scale
        images = images * scale + bias
        images = tf.clip_by_value(images, range_out[0], range_out[1])
        images = tf.cast(images, dtype=out_dtype)
        return images

    def random_flip_left_right(self, images):
        s = tf.shape(images)
        mask = tf.random.uniform([1, 1, 1], 0.0, 1.0)
        mask = tf.tile(mask, [s[0], s[1], s[2]]) # [h, w, c]
        images = tf.where(mask < 0.5, images, tf.reverse(images, axis=[1]))
        return images

    def preprocess_fit_train_image(self, images):
        images = self.adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
        images = self.random_flip_left_right(images)
        return images

    def sample_timesteps(self, n):
        return tf.random.uniform(shape=[n], minval=0, maxval=self.timesteps, dtype=tf.int32)

    def extract(self, x, t):
        return tf.gather(x, t)[:, None, None, None]

    def noise_images(self, x, t): # forward process q
        self.beta = tf.cast(tf.linspace(self.beta_start, self.beta_end, self.timesteps), tf.float32)
        self.alpha = 1 - self.beta
        self.alpha_hat = tf.math.cumprod(self.alpha, axis=0)
        sqrt_alpha_hat = tf.sqrt(tf.gather(self.alpha_hat, t)[:, None, None, None])
        sqrt_one_minus_alpha_hat = tf.sqrt(1 - tf.gather(self.alpha_hat, t)[:, None, None, None])

        eps = tf.random.normal(shape=x.shape)
        x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps

        return x, eps

    def image_processing(self, filename):
        x = tf.io.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=3, dct_method='INTEGER_ACCURATE')
        img = tf.image.resize(x_decode, [self.img_size, self.img_size], align_corners=True, method=tf.image.ResizeMethod.BICUBIC)
        img = self.preprocess_fit_train_image(img)
        t = self.sample_timesteps(n=1)
        x_t, noise = self.noise_images(img, t)
        x_noise = tf.squeeze(x_t)
        return [x_noise, t, noise]

    # For Stable Diffusion Image Process
    def stable_diffusion_processing(self):
        MAX_TEXT_LEN = 77
        image_size = 512
        n_h = image_size // 8
        n_w = image_size // 8
        img = tf.random.normal((n_h, n_w, 4))
        t = self.sample_timesteps(n=1)
        x_t, noise = self.noise_images(img, t)
        x_noise = tf.squeeze(x_t)
        word_ids = tf.random.uniform(shape=[MAX_TEXT_LEN], minval=0, maxval=49407, dtype=tf.int32)
        pos_ids  = tf.range(MAX_TEXT_LEN)
        return [word_ids, pos_ids, x_noise, t, noise]

    def get_input_fn(self):
        def fake_input_fn():
            # DNABert Input Data
            if self.model_type == "dnabert" :
                tf.logging.info("************ Dnabert Input ***********************")
                pkl_file = open('./dataset.pkl', 'rb')
                dataset_list = pickle.load(pkl_file)
                pkl_file.close()
                for i in range(len(dataset_list)):
                    if len(dataset_list[i][0]) < squence_length :
                        append_input = [0] * (squence_length - len(dataset_list[i][0]))
                        append_label = [-100] * (squence_length - len(dataset_list[i][1]))
                        dataset_list[i][0] = np.append(dataset_list[i][0], append_input)
                        dataset_list[i][1] = np.append(dataset_list[i][1], append_label)
                        useful_labels = np.array([0 if label < 0 else 1 for label in dataset_list[i][1]])
                        dataset_list[i][1] = np.array([0 if label < 0 else label for label in dataset_list[i][1]])
                    dataset_list[i] = [tf.constant(dataset_list[i][0]), tf.constant(dataset_list[i][1]), tf.constant(useful_labels, dtype=tf.int64)]
                dataset = tf.data.Dataset.from_tensor_slices(dataset_list).batch(self.batch_size)
                return dataset.prefetch(1)
            elif self.model_type == "unet" :
                # Diffusion Input Data
                tf.logging.info("************ Unet Input ***********************")
                dataset_path = "./data/00000"
                train_images = glob(os.path.join(dataset_path, '*.png')) + glob(os.path.join(dataset_path, '*.jpg'))
                dataset_num = len(train_images)
                image_list = []
                t_list = []
                noise_list = []
                for image in train_images:
                    single_dataset = self.image_processing(image)
                    image_list.append(single_dataset[0])
                    t_list.append(single_dataset[1])
                    noise_list.append(single_dataset[2])
                train_datasets = {"images_noise":image_list, "time":t_list, "noise":noise_list}
                dataset_slice = tf.data.Dataset.from_tensor_slices(train_datasets)
                dataset_iter = dataset_slice.shuffle(buffer_size=dataset_num, reshuffle_each_iteration=True).repeat()
                # dataset_iter = dataset_iter.map(map_func=self.image_processing).batch(self.batch_size, drop_remainder=True)
                dataset_iter = dataset_iter.batch(self.batch_size, drop_remainder=True)
                return dataset_iter.prefetch(1)
            elif self.model_type == "stable_diffusion" :
                # Stable Diffusion Input Data
                tf.logging.info("************ Stable Diffusion Input ***********************")
                dataset_num = 1000
                word_list = []
                pos_list = []
                image_list = []
                t_list = []
                noise_list = []
                for i in range(dataset_num):
                    single_dataset = self.stable_diffusion_processing()
                    word_list.append(single_dataset[0])
                    pos_list.append(single_dataset[1])
                    image_list.append(single_dataset[2])
                    t_list.append(single_dataset[3])
                    noise_list.append(single_dataset[4])
                train_datasets = {"word_ids":word_list, "pos_ids":pos_list, "images_noise":image_list, "time":t_list, "noise":noise_list}
                dataset_slice = tf.data.Dataset.from_tensor_slices(train_datasets)
                dataset_iter = dataset_slice.shuffle(buffer_size=dataset_num, reshuffle_each_iteration=True).repeat()
                # dataset_iter = dataset_iter.map(map_func=self.image_processing).batch(self.batch_size, drop_remainder=True)
                dataset_iter = dataset_iter.batch(self.batch_size, drop_remainder=True)
                return dataset_iter.prefetch(1)
            else :
            # Other Fake data
                dict_data = {}
                for name, feature in self.input_tensors.items():
                    batch_shape = [8] + feature.shape
                    np.random.seed(123123)
                    if feature.dtype == tf.float32:
                        rnd_inputs = tf.constant(np.random.uniform(0, 1, batch_shape).astype(dtype=np.float32))
                        dict_data[name] = rnd_inputs
                    elif feature.dtype == tf.int64:
                        rnd_inputs = tf.constant(np.random.randint(0, 10, batch_shape).astype(dtype=np.int64))
                        dict_data[name] = rnd_inputs
                dataset = tf.data.Dataset.from_tensor_slices(dict_data).repeat(10000).batch(self.batch_size)
                return dataset.prefetch(1)

        def input_fn():
            tf.logging.info("***********num_epochs In Reader is {}***********".format(10000))
            if self.is_training:
                d = tf.data.Dataset.from_tensor_slices(tf.constant(self.files))
                d = d.repeat(10000)
                d = d.shuffle(buffer_size=len(self.input_fps), seed=123123)
                d = tf.data.TFRecordDataset(d)
                d = d.shuffle(buffer_size=self.shuffle_buffer_size, seed=123123)

            else:
                d = tf.data.TFRecordDataset(self.input_fps)
                # Since we evaluate for a fixed number of steps we don't want to encounter
                # out-of-range exceptions.
                d = d.repeat(10000)

            d = self._map_batch_prefetch(d, self._decode_tfrecord)
            return d

        return fake_input_fn if self.fake_input else input_fn
