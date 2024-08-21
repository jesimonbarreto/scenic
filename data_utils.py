"""Util functions for preparing dataset wrapper in scenic."""
import functools
import jax.numpy as jnp
from scenic.dataset_lib import dataset_utils
import tensorflow as tf


# Função para filtrar as classes desejadas
def filter_classes(example, desired_classes):
    return jnp.any(jnp.equal(example['label'], desired_classes))

def filter_classes_ts(dados, allowed_labels=tf.constant([0, 1, 2]), key='label'):
    label = dados[key]
    isallowed = tf.equal(allowed_labels, tf.cast(label, allowed_labels.dtype))
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))

def get_data(
    dataset,
    split,
    batch_size,
    filter_fn=None,
    preprocess_fn=lambda x: x,
    repeats=None,
    shuffle_buffer_size=None,
    prefetch=2,
    cache='loaded',
    repeat_after_batching=False,
    drop_remainder=True,
    data_dir=None,
    ignore_errors=False,
    shuffle_files=True,
    dataset_service_address=None,
):
  """API kept for backwards compatibility."""
  dataset = dataset_utils.get_dataset_tfds(
      dataset=dataset,
      split=split,
      shuffle_files=shuffle_files,
      data_dir=data_dir,
  )
  if 'train' not in split:
    dataset_service_address = None
  if filter_fn:
    dataset = dataset.filter(filter_fn)
  return dataset_utils.make_pipeline(
      data=dataset,
      preprocess_fn=preprocess_fn,
      batch_size=batch_size,
      drop_remainder=drop_remainder,
      cache=cache,
      repeats=repeats,
      prefetch=prefetch,
      shuffle_buffer_size=shuffle_buffer_size,
      repeat_after_batching=repeat_after_batching,
      ignore_errors=ignore_errors,
      dataset_service_address=dataset_service_address,
  )


'''
dataset = datasets['train']

def predicate(x, allowed_labels=tf.constant([0, 1, 2])):
    label = x['label']
    isallowed = tf.equal(allowed_labels, tf.cast(label, allowed_labels.dtype))
    reduced = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced, tf.constant(0.))

dataset = dataset.filter(predicate).batch(20)

for i, x in enumerate(tfds.as_numpy(dataset)):
    print(x['label'])
# [1 0 0 1 2 1 1 2 1 0 0 1 2 0 1 0 2 2 0 1]
# [1 0 2 2 0 2 1 2 1 2 2 2 0 2 0 2 1 2 1 1]
# [2 1 2 1 0 1 1 0 1 2 2 0 2 0 1 0 0 0 0 0]
'''