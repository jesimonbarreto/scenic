"""Data generators for the universal embedding datasets."""

import collections
import functools
import os
from typing import Optional, Union, List

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic.dataset_lib import dataset_utils
import tensorflow as tf
from tensorflow.io import gfile
from scenic.train_lib import train_utils
import ops  # pylint: disable=unused-import
import tensorflow_datasets as tfds

import json

import numpy as np
from collections import OrderedDict


PRNGKey = jnp.ndarray

IMAGE_RESIZE = 256
IMAGE_SIZE = 224

MEAN_RGB = [0.5,0.5,0.5]
STDDEV_RGB = [0.5,0.5,0.5]


"""Data generators for a LOCA dataset."""

import functools
from typing import Optional

from absl import logging
from flax import jax_utils
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.dataset_lib.big_transfer import builder


@datasets.add_dataset('eval_dataset')
def get_dataset(*,
                batch_size,
                eval_batch_size,
                num_shards,
                dtype_str='float32',
                shuffle_seed=0,
                rng=None,
                prefetch_buffer_size=2,
                dataset_configs=None,
                dataset_service_address: Optional[str] = None):
  """Returns a generator for training Dino on a specified dataset.

  Args:
    batch_size: int; Determines the training batch size.
    eval_batch_size: int; Not used.
    num_shards: int;  Number of shards --> batch shape: [num_shards, bs, ...].
    dtype_str: Data type of the image (e.g. 'float32').
    shuffle_seed: int; Seed for shuffling the training data.
    rng: Not used.
    prefetch_buffer_size: int; Buffer size for the device prefetch.
    dataset_configs: dict; Dataset specific configurations.
    dataset_service_address: If set, will distribute the training dataset using
      the given tf.data service at the given address.

  Returns:
    A dataset_utils.Dataset() which includes train_iter and dict of meta_data.
  """
  del eval_batch_size, rng
  logging.info('Loading train split of the %s for eval training.',
               dataset_configs.dataset)
  #n_train_ex = dataset_utils.get_num_examples(dataset_configs.dataset,
  #                                            dataset_configs.train_split)

  tfds.load(
    dataset_configs.dataset,
    data_dir= dataset_configs.get('dataset_dir'),
    download = True
  )

  train_ds = dataset_utils.get_data(
      dataset=dataset_configs.dataset,
      split=dataset_configs.train_split,
      data_dir=dataset_configs.get('dataset_dir'),
      batch_size=dataset_configs.batch_size_train,
      preprocess_fn=builder.get_preprocess_fn(dataset_configs.pp_train),
      shuffle_buffer_size=dataset_configs.shuffle_buffer_size,
      prefetch=dataset_configs.get('prefetch_to_host', 2),
      drop_remainder=True,
      cache=False,
      ignore_errors=True)
  
  eval_ds = dataset_utils.get_data(
      dataset=dataset_configs.dataset,
      split=dataset_configs.test_split,
      data_dir=dataset_configs.get('dataset_dir'),
      batch_size=dataset_configs.batch_size_test,
      preprocess_fn=builder.get_preprocess_fn(dataset_configs.pp_train),
      shuffle_buffer_size=dataset_configs.shuffle_buffer_size,
      prefetch=dataset_configs.get('prefetch_to_host', 2),
      drop_remainder=True,
      cache=False,
      ignore_errors=True)

  if dataset_service_address:
    if shuffle_seed is not None:
      raise ValueError('Using dataset service with a random seed causes each '
                       'worker to produce exactly the same data. Add '
                       'config.shuffle_seed = None to your config if you '
                       'want to run with dataset service.')
    logging.info('Using the tf.data service at %s', dataset_service_address)
    assert dataset_configs.shuffle_buffer_size is not None
    train_ds = dataset_utils.distribute(train_ds, dataset_service_address)
    eval_ds = dataset_utils.distribute(eval_ds, dataset_service_address)

  n_train_ex = dataset_utils.get_num_examples(dataset_configs.dataset,
                                              dataset_configs.train_split)
  shard_batches = functools.partial(dataset_utils.shard, n_devices=num_shards)
  train_iter = iter(train_ds)
  train_iter = map(dataset_utils.tf_to_numpy, train_iter)
  train_iter = map(shard_batches, train_iter)
  train_iter = jax_utils.prefetch_to_device(train_iter, prefetch_buffer_size)

  eval_iter = iter(eval_ds)
  eval_iter = map(dataset_utils.tf_to_numpy, eval_iter)
  eval_iter = map(shard_batches, eval_iter)
  eval_iter = jax_utils.prefetch_to_device(eval_iter, prefetch_buffer_size)

  image_size = eval_ds.element_spec['image_resized'].shape
  labels_size = train_ds.element_spec['label_onehot']
  logging.info(f' train {train_ds.element_spec.keys()}')
  logging.info(f' test {eval_ds.element_spec.keys()}')
  logging.info(f' image {image_size}')
  logging.info(f' labels {labels_size}')

  input_shape = (-1,) + tuple(train_ds.element_spec['image_resized'].shape[1:])
  labels_size = train_ds.element_spec['label_onehot'].shape
  logging.info('input shape details %s', input_shape)
  logging.info('samples details %s', labels_size)

  meta_data = {
      'input_shape': input_shape,
      'num_train_examples': n_train_ex,
      'input_dtype': getattr(jnp, dtype_str),
      'label_data' : labels_size,
  }
  return dataset_utils.Dataset(train_iter, eval_iter, None, meta_data)



def _normalize_image(
    image,
    normalization_statistics,
):

  if normalization_statistics is None:

      image /= tf.constant(255, shape=[1, 1, 3], dtype=image.dtype)
      image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
      image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)

  else:

      image /= tf.constant(255, shape=[1, 1, 3], dtype=image.dtype)
      image -= tf.constant(normalization_statistics["MEAN_RGB"], shape=[1, 1, 3], dtype=image.dtype)
      image /= tf.constant(normalization_statistics["STDDEV_RGB"], shape=[1, 1, 3], dtype=image.dtype)


  return image


def _process_train_split(image):

  # Resize to 256x256
  image = _resize(image, IMAGE_RESIZE)
  # Random crop to 224x224
  image = tf.image.random_crop(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  # Random flip
  image = tf.image.random_flip_left_right(image)
  return image


def _process_test_split(image): 
  
  # Resize the small edge to 224.
  image, new_size = _resize_smaller_edge(image, IMAGE_SIZE)
  # Central crop to 224x224.
  h, w = new_size
  if h > w:
    image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
  else:
    image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)

  return image


def _resize(image, image_size):
  
  """
  Resizes the image.

  Args:
    image: Tensor; Input image.
    image_size: int; Image size.

  Returns:
    Resized image.
  """
  
  return tf.image.resize(
      image, 
      [image_size, image_size], 
      method=tf.image.ResizeMethod.BILINEAR,
  )


def _resize_smaller_edge(
  image, 
  image_size,
):
  
  """Resizes the smaller edge to the desired size and keeps the aspect ratio."""
  
  shape = tf.shape(image)
  height, width = shape[0], shape[1]
  if height <= width:
    # Resize to [224, width / height * 224]
    new_height = image_size
    new_width = tf.cast((width / height) * image_size, tf.int32)
  else:
    # Resize to [height / width *224, 224]
    new_width = image_size
    new_height = tf.cast((height / width) * image_size, tf.int32)

  return tf.image.resize(
      image, 
      [new_height, new_width], 
      method=tf.image.ResizeMethod.BILINEAR,
  ),(new_height, new_width)



def preprocess_example(
  example, 
  split,
  total_classes,
  domain, 
  augment=False, 
  dtype=tf.float32, 
  label_offset=0,
  domain_mask_range = None,
  domain_idx = -1,
  normalization_statistics = None,
):
  """Preprocesses the given image.

  Args:
    example: The proto of the current example.
    split: str; One of 'train' or 'test'.
    domain: int; the domain of the dataset.
    augment: whether to augment the image.
    dtype: Tensorflow data type; Data type of the image.
    label_offset: int; The offset of label id.

  Returns:
    A preprocessed image `Tensor`.
  """
  
  features = tf.io.parse_single_example(
      example,
      features={
        'image_bytes': tf.io.FixedLenFeature([], tf.string),
        'key': tf.io.FixedLenFeature([], tf.string),
        'class_id': tf.io.FixedLenFeature([], tf.int64),
      },
  )
  
  image = tf.io.decode_jpeg(features['image_bytes'], channels=3)

  if split == 'train' and augment:
    image = _process_train_split(image)
  else:
    image = _process_test_split(image)

  image = _normalize_image(image,normalization_statistics)
  image = tf.image.convert_image_dtype(image, dtype=dtype)
  
  domain_mask = np.full((total_classes),False)
  domain_mask[domain_mask_range[0]:domain_mask_range[1]] = True

  return {
    'inputs': image,
    'label': features['class_id'] + label_offset,
    'domain': domain,
    'domain_mask': domain_mask,
    'domain_idx': domain_idx,
  }


def preprocess_example_eval(
  example, 
  split,
  total_classes,
  domain, 
  augment=False, 
  dtype=tf.float32, 
  domain_mask_range = None,
  domain_idx = -1,
  normalization_statistics = None,
):
  """Preprocesses the given image.

  Args:
    example: The proto of the current example.
    split: str; One of 'train' or 'test'.
    domain: int; the domain of the dataset.
    augment: whether to augment the image.
    dtype: Tensorflow data type; Data type of the image.

  Returns:
    A preprocessed image `Tensor`.
  """
  
  features = tf.io.parse_single_example(
      example,
      features={
        'image_bytes': tf.io.FixedLenFeature([], tf.string),
        'key': tf.io.FixedLenFeature([], tf.string),
      },
  )
  
  image = tf.io.decode_jpeg(features['image_bytes'], channels=3)

  if split == 'train' and augment:
    image = _process_train_split(image)
  else:
    image = _process_test_split(image)

  image = _normalize_image(image,normalization_statistics)
  image = tf.image.convert_image_dtype(image, dtype=dtype)

  domain_mask = np.full((total_classes),False)
  domain_mask[domain_mask_range[0]:domain_mask_range[1]] = True

  return {
    'inputs': image,
    'domain': domain,
    'domain_mask': domain_mask,
    'domain_idx': domain_idx,
  }




def build_dataset_new(
  dataset_fn,
  batch_size=None,
  shuffle_buffer_size=256,
  seed=None,
  repeat=False,
  sampling=None,
  knn = False,
  **dataset_kwargs,
):
  """Dataset builder that takes care of strategy, batching and shuffling.

  Args:
    dataset_fn: function; A function that loads the dataset.
    batch_size: int; Size of the batch.
    shuffle_buffer_size: int; Size of the buffer for used for shuffling.
    seed: int; Random seed used for shuffling.
    repeat: bool; Whether to repeat the dataset.
    sampling: str; the sampling option for multiple datasets
    **dataset_kwargs: dict; Arguments passed to TFDS.

  Returns:
    Dataset.
  """

  dataset_kwargs['knn'] = knn

  def _shuffle_batch_prefetch(dataset, replica_batch_size, split):
    
    if split == 'train' and repeat:
    
      dataset = dataset.shuffle(
          shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True,
      )
      dataset = dataset.batch(replica_batch_size, drop_remainder=True)
    
      #shuffle the batches again
      batch_shuffle_buffer_size = 16

      dataset = dataset.shuffle(
          batch_shuffle_buffer_size, seed=seed, reshuffle_each_iteration=True,
      )

    else:

      #knn case
      dataset = dataset.batch(replica_batch_size, drop_remainder=False)

    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    dataset = dataset.with_options(options)
    
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)


  def _dataset_fn(input_context=None):
    """Dataset function."""

    replica_batch_size = batch_size

    if input_context:
      replica_batch_size = input_context.get_per_replica_batch_size(batch_size)

    #dataset_fn here is "build_universal_embedding_dataset_new"
    ds_dict = dataset_fn(**dataset_kwargs)

    split = dataset_kwargs.get('split')
    
    if split == 'train' and repeat:

      for ds_name,ds in ds_dict.items():
        ds = ds.repeat()
        ds_dict[ds_name] = _shuffle_batch_prefetch(ds, replica_batch_size, split)

      return ds_dict
    
    else:

      #case of knn dataset (only one domain at a a time)
      assert len(list(ds_dict.keys())) == 1
      domain_name = list(ds_dict.keys())[0]

      return _shuffle_batch_prefetch(ds_dict[domain_name], replica_batch_size, split)

  return _dataset_fn()



def get_training_dataset_new(
  config: ml_collections.ConfigDict,
  num_local_shards: Optional[int] = None,
  prefetch_buffer_size: Optional[int] = 2,
  dataset_configs: Optional[ml_collections.ConfigDict] = None,
):
  """Returns generators for the universal embedding train, validation, and test sets.

  Args:
    config: The configuration of the experiment.
    data_rng: Random number generator key to use for the dataset.
    num_local_shards: Number of shards for each batch. So (bs, ...) becomes
      (num_local_shards, bs//num_local_shards, ...). If not specified, it will
      be number of local devices.
    prefetch_buffer_size: int; Buffer size for the device prefetch.
    dataset_configs: Configuration of the dataset, if not reading directly from
      the config.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """
  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())

  base_dir = config.train_dataset_dir

  dataset_name = config.dataset_name
  dataset_names = dataset_name.split(',')
  batch_size = config.batch_size

  if batch_size % device_count > 0:
    raise ValueError(
        f'Batch size ({batch_size}) must be divisible by the '
        f'number of devices ({device_count})'
    )

  local_batch_size = batch_size // jax.process_count()
  logging.info('local_batch_size : %d', local_batch_size)

  eval_batch_size = config.get('eval_batch_size', batch_size)
  local_eval_batch_size = eval_batch_size // jax.process_count()
  logging.info('local_eval_batch_size : %d', local_eval_batch_size)

  shuffle_seed = config.get('shuffle_seed', None)

  dataset_configs = dataset_configs or config.get('dataset_configs', {})
  num_local_shards = num_local_shards or jax.local_device_count()
  logging.info('local_eval_batch_size : %d', local_eval_batch_size)

  # use different seed for each host
  if shuffle_seed is None:
    local_seed = None
  else:
    data_seed = 0
    local_seed = data_seed + jax.process_index()

  maybe_pad_batches_train = functools.partial(
      dataset_utils.maybe_pad_batch,
      train=True,
      batch_size=local_batch_size,
  )

  shard_batches = functools.partial(
      dataset_utils.shard, n_devices=num_local_shards
  )

  train_ds_dict = build_dataset_new(
    dataset_fn=build_universal_embedding_dataset_new,
    dataset_names=dataset_names,
    split='train',
    batch_size=local_batch_size,
    seed=local_seed,
    augment=True,
    repeat=True,
    sampling=dataset_configs.get('sampling', None),
    base_dir = base_dir,
    config = config,
  )

  train_iter_dict = {}

  for ds_name,ds in train_ds_dict.items():
    train_iter_dict[ds_name] = build_ds_iter(
      train_ds_dict[ds_name], maybe_pad_batches_train, shard_batches, prefetch_buffer_size
  )

  input_shape = (
    -1,
    IMAGE_SIZE,
    IMAGE_SIZE,
    3,
  )

  num_train_examples, num_test_examples, num_val_examples, num_classes = (
    0,
    0,
    0,
    0,
  )

  dataset_samples = OrderedDict()

  for name in dataset_names:
    num_train_examples += DATASET_INFO[name]['num_train_examples']
    num_test_examples += DATASET_INFO[name]['num_test_examples']
    num_val_examples += DATASET_INFO[name]['num_val_examples']
    num_classes += DATASET_INFO[name]['num_train_classes']
    dataset_samples[name] = DATASET_INFO[name]['num_train_examples']


  domain_indices = [DATASET_INFO[dat_name]["domain"] for dat_name in dataset_names]

  meta_data = {
    'dataset_name': dataset_name,
    'domain_indices': domain_indices,
    'num_classes': num_classes,
    'input_shape': input_shape,
    'num_train_examples': num_train_examples,
    'num_test_examples': num_test_examples,
    'num_val_examples': num_val_examples,
    'input_dtype': getattr(jnp, config.data_dtype_str),
    'target_is_onehot': False,
    'dataset_samples': dataset_samples,
  }

  return UniversalEmbeddingTrainingDataset(
      train_iter_dict,
      meta_data,
  )



def dataset_lookup_key(dataset_name, split):
  return dataset_name + ':' + split



def get_knn_eval_datasets(
    config,
    base_dir,
    data_rng,
    dataset_service_address,
    dataset_names: Union[List[str], str],
    eval_batch_size: int,
):
  """Returns generators for the universal embedding train, validation, and test sets.

  Args:
    dataset_names: a lsit of dataset names.
    eval_batch_size: The eval batch size.
    prefetch_buffer_size: int; Buffer size for the device prefetch.

  Returns:
    A dataset_utils.Dataset() which includes a train_iter, a valid_iter,
      a test_iter, and a dict of meta_data.
  """

  base_dir = config.eval_dataset_dir

  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())

  local_eval_batch_size = eval_batch_size // jax.process_count()

  logging.info('local_eval_batch_size : %d', local_eval_batch_size)

  num_local_shards = jax.local_device_count()

  if isinstance(dataset_names, str):
    dataset_names = dataset_names.split(',')

  knn_info, knn_setup, size_info = {}, {}, {}

  knn_info['json_data'] = {}

  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=dataset_service_address) 

  
  meta_data = {
    'dataset_names': ','.join(dataset_names),
    'top_k': int(config.top_k),
    'size_info': size_info,
  }

  return dataset, meta_data
