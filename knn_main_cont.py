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
import ops  # pylint: disable=unused-import
from jax.nn import softmax

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
from scenic.dataset_lib import dataset_utils

import dino_dataset  # pylint: disable=unused-import
#from scenic.dataset_lib import tinyImagenet_dataset
import datasets_eval
import optax
from scenic.train_lib import lr_schedules
import copy

import functools
from typing import Any, Callable, Dict, Tuple, Optional, Type
import flax
from flax import jax_utils
from flax import linen as nn
from jax import vmap
from jax.lax import map as map_

from functools import partial
from jax import jit

import matplotlib.pyplot as plt


FLAGS = flags.FLAGS


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[
    [jnp.ndarray, Dict[str, jnp.ndarray]], Dict[str, Tuple[float, int]]
]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]
LrFn = Callable[[jnp.ndarray], jnp.ndarray]

def normalize(input, p=2.0, axis=1, eps=1e-12):
    norms = jnp.linalg.norm(input, ord=p, axis=axis, keepdims=True)
    return input / jnp.maximum(norms, eps)

def representation_fn_eval(
    train_state: train_utils.TrainState,
    batch: Batch,
    *,
    flax_model: nn.Module,
    project_feats = True,
    gather_to_host: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Feeds the inputs to the model and returns their representations.

  Args:
    train_state: TrainState, the state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data from the dataset.
    flax_model: A Flax model.
    gather_to_host: Whether to gather results from all devices to the host,
      rather than leaving them distributed.

  Returns:
    Representation learned by the model for the given inputs and the labels and
    masks. If `gather_to_host` is True, these are collected from all hosts.
  """
  #variables = {'params': train_state.params, **train_state.model_state}

  '''embedding = flax_model.apply(
    variables, 
    batch['sample'],
    train=False,
    return_feats = True,
    debug=False, 
    project_feats = project_feats,
  )'''
  embedding = flax_model.apply(
        {'params': train_state.params},
        batch['image_resized'],
        seqlen=-1,
        seqlen_selection='consecutive',
        drop_moment='late',
        backbone = True,
        train=False)
  embedding = jnp.squeeze(embedding['x_norm_clstoken'])
  embedding = normalize(embedding)

  if gather_to_host:
    embedding = jax.lax.all_gather(embedding, 'batch')
    batch = jax.lax.all_gather(batch, 'batch')
  

  return embedding

def knn_evaluate(
  rng: jnp.ndarray,
  config: ml_collections.ConfigDict,
  workdir: str,
  writer: metric_writers.MetricWriter,
) -> None:

  lead_host = jax.process_index() == 0

  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  
  train(
      rng=rng,
      config=config,
      dataset=dataset,
      workdir=workdir,
      writer=writer)
  
def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[Any, Any]:

  # Inicializa um array para armazenar as contagens combinadas
  total_counts_val = jnp.zeros(1000, dtype=int)
  

  for i in range(config.steps_per_epoch_eval):
    print(f'Processing {i}')
    batch_eval = next(dataset.valid_iter)
    shp = batch_eval['label'].shape
    print(f'shape before {shp}')
    label_eval = batch_eval['label'].reshape((shp[0]*shp[1]))
    counts = jnp.bincount(label_eval, minlength=1000)
    print(f'shape after {label_eval.shape}')
    total_counts_val += counts
    print(f'counts {counts}')

  combined_value_counts_val = {i: total_counts_val[i] for i in range(1000)}

    
  jnp.savez('/home/jesimonbarreto/count_samples_imgnet.npz', val_arr=total_counts_val, val_dic=combined_value_counts_val)

  print('Finished')



if __name__ == '__main__':
  app.run(main=knn_evaluate)