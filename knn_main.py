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

import classification_with_knn_eval_trainer
import datasets
import knn_utils

FLAGS = flags.FLAGS

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
  

  #project feats or not
  representation_fn_knn = functools.partial(
    classification_with_knn_eval_trainer.representation_fn_eval,
    flax_model = model.flax_model, 
    project_feats = config.project_feats_knn
  )

  knn_eval_batch_size = config.get('knn_eval_batch_size') or config.batch_size

  repr_fn = jax.pmap(
        representation_fn_knn, 
        donate_argnums=(1,), 
        axis_name='batch',
  )

  train_dir = config.get('train_dir')

  for epoch in range(config.knn_start_epoch,config.knn_end_epoch+1):

    step = epoch * config.steps_per_epoch

    print(f"step: {step}")

    if not config.preextracted:
      ckpt_file = os.path.join(train_dir,str(step))  
      ckpt_info = ckpt_file.split('/')
      ckpt_dir = '/'.join(ckpt_info[:-1])
      ckpt_num = ckpt_info[-1].split('_')[-1]

      try:

        train_state, _ = train_utils.restore_checkpoint(
          ckpt_dir, 
          train_state, 
          assert_exist=True, 
          step=int(ckpt_num),
        )
        
      except:

        sys.exit("no checkpoint found")

      train_state = jax_utils.replicate(train_state)

    else:

      train_state = None


    # extract features
    @jax.jit
    def extract_features(batch):
      features = repr_fn(train_state, batch)
      return features  # Return extracted features for the batch
      
    for batch in next(dataset.train_iter):
      batch['emb'] = extract_features(batch)

    for batch in next(dataset.eval_iter):
      batch['emb'] = extract_features(batch)
    
    @jax.vmap
    def euclidean_distance(x1, x2):
      return jnp.linalg.norm(x1 - x2, axis=-1)

    @jax.vmap
    def cosine_similarity(x1, x2):
      return jnp.dot(x1, x2) / (jnp.linalg.norm(x1, axis=-1) * jnp.linalg.norm(x2, axis=-1))
    
    print('Dataset keys')
    print(dataset.keys())
    len_test = 0
    correct_pred = 0
    for batch_eval in next(dataset.eval_iter):
      dist_all = []
      labels = []
      len_test += len(batch_eval)
      for batch_train in next(dataset.train_iter):
        dist_ = jax.vmap(euclidean_distance, in_axes=(0, 1))(batch_eval['emb'], batch_train['emb'])
        dist_all.append(dist_)
        labels.append(batch_train['labels'])
      dist_all = jnp.concatenate(dist_all)
      labels = jnp.concatenate(labels)
      @jax.vmap
      def knn_vote(k, distances, train_labels):
          # Get k nearest neighbors for each test sample
          nearest_indices = jnp.argpartition(distances, k - 1, axis=-1)[:k]
          # Count occurrences of each class among neighbors
          class_counts = jnp.bincount(train_labels[nearest_indices.flatten()], axis=1)
          # Predict class with the highest vote count
          return jnp.argmax(class_counts, axis=-1)

      predictions = knn_vote(k=5, distances=dist_all, train_labels=labels)
    
      # Compare predictions with actual test labels
      correct_predictions = jnp.equal(predictions, dataset.labels)
      correct_pred += jnp.sum(correct_predictions)

    # Calculate accuracy as the ratio of correct predictions to total test samples
    accuracy = correct_pred / len_test

    print(f"Number of correct predictions: {correct_pred}")
    print(f"Accuracy: {accuracy:.4f}")
    

  train_utils.barrier_across_hosts()



if __name__ == '__main__':
  app.run(main=knn_evaluate)