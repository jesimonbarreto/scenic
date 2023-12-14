"""Script for Knn evalulation."""
import functools

from clu import metric_writers
from absl import flags
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.train_lib import train_utils

import os
import sys

if sys.version_info.major == 3 and sys.version_info.minor >= 10:

  from collections.abc import MutableMapping
else:
  from collections import MutableMapping

import vit_dino as vit
import utils_dino as utils
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
import datasets
import dino_dataset  # pylint: disable=unused-import
from scenic.dataset_lib import tinyImagenet_dataset


FLAGS = flags.FLAGS

def get_datasets(batch=3):
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=batch))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=batch))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds

def compute_diff(u, v):
    return (u[:, None] - v[None, :]) ** 2

# We pmap the built-in argsort function along the first axes.
# We sort a matrix with shape (devices, n_test // devices, n_train)
p_argsort = jax.pmap(jnp.argsort, in_axes=0)

def compute_distance(U, V):
    return compute_diff(U, V).mean(axis=-1)

def compute_k_closest(U, V, k, devices, n_test):
    D = compute_distance(U, V)
    D = D.reshape(devices, n_test // devices, -1)
    nearest = p_argsort(D)[..., 1:k+1]
    return nearest

compute_diff = jax.vmap(compute_diff, in_axes=1, out_axes=-1)

# We pmap the built-in argsort function along the first axes.
# We sort a matrix with shape (devices, n_test // devices, n_train)
p_argsort = jax.pmap(jnp.argsort, in_axes=0)  

n_train = 30_000
n_test = 10 * 8

def knn_evaluate(
  rng: jnp.ndarray,
  config: ml_collections.ConfigDict,
  workdir: str,
  writer: metric_writers.MetricWriter,
) -> None:

  lead_host = jax.process_index() == 0

  k=20

  devices = 8
  train, test = get_datasets(batch=-1)

  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  # Build the loss_fn, metrics, and flax_model.
  model = vit.ViTDinoModel(config, dataset.meta_data)


  # Randomly initialize model parameters.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config, rngs=init_rng)

  rng, init_rng = jax.random.split(rng)

  #Load model
  train_state = train_utils.TrainState(
    params=params, 
    model_state=model_state
  )
  #chrono = train_utils.Chrono()
  if config.checkpoint:
    train_state, start_step = utils.restore_checkpoint(workdir, train_state)
  #chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})
  # Replicate the training state: optimizer, params and rng.
  train_state = jax_utils.replicate(train_state)
  del params


  
  predictions = []
  for data_point in dataset['batch']:
    _, pred_ = model.apply(
        {'params': train_state.params},
        data_point)
    predictions.append(pred_)
  # Concatenate individual predictions into a single array
  X_train = jnp.concatenate(predictions)


  '''predictions = []
  for data_point in test.get:
    _, pred_ = model.apply(
        {'params': train_state.params},
        data_point)
    predictions.append(pred_)
  # Concatenate individual predictions into a single array
  X_test = jnp.concatenate(predictions)'''

  y_train = dataset['label']
  y_test = dataset['label']
  print(X_train.shape)
  print(X_test.shape)
  
  k_nearest = compute_k_closest(X_test, X_train, k)
  if jax.process_index() == 0:
    print(k_nearest.shape)
    k_nearest = k_nearest.reshape(-1, k)
    class_rate = (y_train[k_nearest, ...].mean(axis=1).round() == y_test).mean()
    print(f"{class_rate=}")


  train_utils.barrier_across_hosts()



if __name__ == '__main__':
  app.run(main=knn_evaluate)