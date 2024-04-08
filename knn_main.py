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
from scenic.dataset_lib import tinyImagenet_dataset
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

FLAGS = flags.FLAGS


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricFn = Callable[
    [jnp.ndarray, Dict[str, jnp.ndarray]], Dict[str, Tuple[float, int]]
]
LossFn = Callable[[jnp.ndarray, Batch, Optional[jnp.ndarray]], float]
LrFn = Callable[[jnp.ndarray], jnp.ndarray]

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
        batch['image'],
        seqlen=-1,
        seqlen_selection='consecutive',
        drop_moment='late',
        backbone = True,
        train=False)
  
  #embedding = jnp.mean(embedding, axis=1)
  
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

  # Only one model function but two sets of parameters.
  ema_params = copy.deepcopy(params)

  # Get learning rate and ema temperature schedulers.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
  momentum_parameter_scheduler = lr_schedules.compound_lr_scheduler(
      config.momentum_rate)

  # Create optimizer.
  weight_decay_mask = jax.tree_map(lambda x: x.ndim != 1, params)
  tx = optax.inject_hyperparams(optax.adamw)(
      learning_rate=learning_rate_fn, weight_decay=config.weight_decay,
      mask=weight_decay_mask,)
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  # Create chrono class to track and store training statistics and metadata.
  chrono = train_utils.Chrono()

  # Create the TrainState to track training state (i.e. params and optimizer).
  train_state = utils.TrainState(
      global_step=0, opt_state=opt_state, tx=tx, params=params,
      ema_params=ema_params, rng=rng, metadata={'chrono': chrono.save()})
  
  start_step = train_state.global_step
  
  if config.checkpoint:
    train_state, start_step = utils.restore_checkpoint(workdir, train_state)
  chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})
  # Replicate the training state: optimizer, params and rng.
  train_state = jax_utils.replicate(train_state)
  del params, ema_params
  

  

  knn_eval_batch_size = config.get('knn_eval_batch_size') or config.batch_size

  train_dir = config.get('train_dir')
  print(f'{train_dir}')
  steps = config.get('steps_checkpoints')
  files_save = config.get('dir_files')
  num_classes = config.get('num_classes')

  for step in steps:

    print(f"step: {step}")
    

    if not config.preextracted:
      ckpt_file = os.path.join(train_dir,'checkpoint_'+str(step))  
      ckpt_info = ckpt_file.split('/')
      ckpt_dir = '/'.join(ckpt_info[:-1])
      ckpt_num = ckpt_info[-1].split('_')[-1]
      print(f"file: {ckpt_file}")
      print(f"ckpt_num: {ckpt_num}")

      #try:

      train_state, _ = train_utils.restore_checkpoint(
          ckpt_dir, 
          train_state, 
          assert_exist=True, 
          step=int(ckpt_num),
        )
        
      #except:

      #  sys.exit("no checkpoint found")
      #  continue

      train_state = jax_utils.replicate(train_state)

    else:

      train_state = None

    #project feats or not
    representation_fn_knn = functools.partial(
      representation_fn_eval,
      flax_model = model.flax_model, 
    )
    repr_fn = jax.pmap(
          representation_fn_knn, 
          donate_argnums=(1,), 
          axis_name='batch',
    )

    # extract features
    @jax.jit
    def extract_features(batch):
      features = repr_fn(train_state, batch)
      return features  # Return extracted features for the batch
    
    print('Starting to extract features')
    for i in range(config.steps_per_epoch):
      batch_train = next(dataset.train_iter)
      emb_train = extract_features(batch_train)
      label_train = batch_train['label'][0]
      f = batch_train['image']
      jnp.savez(files_save+f'{i}', emb=emb_train, label=label_train)

    print('Finishing extract features')
    #print(dataset.meta_data.keys)
    '''for i in range(config.steps_per_epoch):
      print(i)
      batch = next(dataset.train_iter)
      print(batch['image'].shape)
      batch['emb'] = extract_features(batch)

    for i in range(config.steps_per_epoch_eval):
      print(i)
      batch = next(dataset.valid_iter)

      print(batch['image'].shape)
      batch['emb'] = extract_features(batch)'''
    
    '''def my_transformation_function(batch):
      # Existing logic for accessing desired key in batch
      #transformed_data = ...  # Apply your transformation to the key value
      batch["new_key"] = '1'
      return batch

    dataset.train_iter = map_(my_transformation_function, dataset.train_iter)
    dataset.train_iter.keys()'''

    @jax.vmap
    def euclidean_distance(x1, x2):
      return jnp.linalg.norm(x1 - x2, axis=-1)

    @jax.vmap
    def cosine_similarity(x1, x2):
      return jnp.dot(x1, x2) / (jnp.linalg.norm(x1, axis=-1) * jnp.linalg.norm(x2, axis=-1))
    
    def compute_diff(u, v):
      return (u[:, None] - v[None, :]) ** 2

    compute_diff = jax.vmap(compute_diff, in_axes=1, out_axes=-1)

    p_argsort = jax.pmap(jnp.argsort, in_axes=0)


    def compute_distance(U, V):
      return compute_diff(U, V).mean(axis=-1)
    
    devices = jax.device_count()
    n_test = config.dataset_configs.batch_size_test
    
    ks = config.get('ks')
    
    def compute_k_closest(U, V, k):
      D = compute_distance(U, V)
      D = D.reshape(devices, n_test // devices, -1)
      nearest = p_argsort(D)[..., 1:k+1]
      return nearest
    
    for k in ks:
      print(f'K: {k}')
      len_test = 0
      correct_pred = 0
      predicts_acc = []
      for i in range(config.steps_per_epoch_eval):
        print(f'processing step eval {i}')
        batch_eval = next(dataset.valid_iter)
        emb_test = extract_features(batch_eval)
        #print(f'embeeding shape test {emb_test.shape}')
        dist_all = []
        labels = []
        len_test += len(batch_eval)
        for i in range(config.steps_per_epoch):
          #batch_train = next(dataset.train_iter)
          data_load = jnp.load(files_save+f'{i}.npz')
          emb_train = data_load['emb']#extract_features(batch_train)
          label_train = data_load['label']#batch_train['label'][0]
          #f = batch_train['image_resized']
          #print(f'batch shape train {i}: {f.shape}')
          #print(f'embeeding shape train {i}: {emb_train[0].shape}')
          #print(f'embeeding shape test {i}: {emb_test[0].shape}')
          #print(batch_train['label'].shape)
          #dist_ = jax.vmap(euclidean_distance, in_axes=(0, 1))(emb_test, emb_train)
          dist_ = compute_distance(emb_test[0][0], emb_train[0][0])
          
          #print(f'dist shape train {i}: {dist_.shape} {dist_[0]}')
          #print(f'labels shape train {i}: {label_train.shape} {label_train[0]}')

          dist_all.append(dist_)
          labels.append(label_train['label'][0])
        dist_all = jnp.concatenate(dist_all, axis=1)
        labels = jnp.concatenate(labels)
        #print(f'shape dist_all ------------ {dist_all.shape}')
        #print(f'shape labels   ------------ {labels.shape}')

        #dist_all = dist_all.reshape(devices, n_test // devices, -1)
        dist_all = jnp.repeat(jnp.expand_dims(dist_all,axis=0), devices, axis=0)
        k_nearest = p_argsort(dist_all)[0][..., 1:k+1]
        
        print(k_nearest.shape)
        k_nearest = k_nearest.reshape(-1, k)
        
        k_nearest_labels = labels[k_nearest]  # Shape: (n, 5)

        #most_repetitive_labels = jnp.apply_along_axis(lambda row: jnp.bincount(jnp.asarray(row)).argmax(), axis=1, arr=jnp.asarray(k_nearest_labels))

        most_repetitive_labels = [(num_classes-1) - jnp.bincount(row, minlength=num_classes)[::-1].argmax() for row in k_nearest_labels]

        comparison = jnp.asarray(most_repetitive_labels) == batch_eval['label'][0]
        accuracy = comparison.mean()  # Proportion of correct matches
        print(f"Accuracy: {accuracy:.4f}")
        predicts_acc.append(accuracy)

        #class_rate = (labels[k_nearest, ...].mean(axis=1).round() == batch_eval['label'][0]).mean()
        #print(f"{class_rate=}")
        
        '''@partial(jit, static_argnums=0)
        def knn_vote(k, distances, train_labels):
            # Get k nearest neighbors for each test sample
            print(f'k {k}')
            print(f'distances shape {distances}')
            print(f'labels shape {train_labels}')
            #k_mat = int(k[0])
            #print(f'k {k_mat}')
            nearest_indices = jnp.argpartition(distances, 5 - 1, axis=-1)[:5]
            # Count occurrences of each class among neighbors
            class_counts = jnp.bincount(train_labels[nearest_indices.flatten()])
            # Predict class with the highest vote count
            return jnp.argmax(class_counts, axis=-1)
        
        print(f'Dist all [0] : {dist_all[0]}')
        k_ = [5]
        
        n = jax.device_count()
        dist_all = jnp.repeat(jnp.array(dist_all).reshape(1,-1), n, axis=0) 
        labels = jnp.repeat(jnp.array(labels).reshape(1,-1), n, axis=0)
        k_ = jnp.repeat(jnp.array(k_).reshape(1,-1), n, axis=0)
        print(f'shape dist_all ------------ {dist_all.shape}')
        print(f'shape labels   ------------ {labels.shape}')

        predictions = vmap(knn_vote)(k=jnp.array(k_), distances=dist_all, train_labels=labels)
      
        # Compare predictions with actual test labels
        print('-----------------------------------')
        print(f' pred  {predictions[0]}')
        labels_eval = batch_eval['label'][0]
        print(f' pred  {labels_eval}')
        correct_predictions = jnp.equal(predictions[0], labels_eval)
        correct_pred += jnp.sum(correct_predictions)'''
      
      predicts_acc = jnp.asarray(predicts_acc)
      result = jnp.mean(predicts_acc)
      
      print(f"{k} Neighborhood: Accuracy total : {result:.4f} ---- executions {predicts_acc.shape} ----- step {step}")
  
    # Calculate accuracy as the ratio of correct predictions to total test samples
    #accuracy = correct_pred / len_test

    #print(f"Number of correct predictions: {correct_pred}")
    #print(f"Number of total predictions: {len_test}")
    #print(f"Accuracy: {accuracy:.4f}")
    

  train_utils.barrier_across_hosts()



if __name__ == '__main__':
  app.run(main=knn_evaluate)