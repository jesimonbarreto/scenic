"""Script for Knn evalulation."""
import functools

from clu import metric_writers

from absl import logging
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

import classification_with_knn_eval_trainer
import datasets
import knn_utils
import models
import vit_dino as vit



def knn_evaluate(
  rng: jnp.ndarray,
  config: ml_collections.ConfigDict,
  workdir: str,
  writer: metric_writers.MetricWriter,
) -> None:

  lead_host = jax.process_index() == 0

  #dataset used for training
  dataset_dict = datasets.get_training_dataset_new(config)

  
  # Build the loss_fn, metrics, and flax_model.
  model = vit.ViTDinoModel(config, dataset_dict.meta_data)


  # Randomly initialize model parameters.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset_dict.meta_data['input_shape'],
                    dataset_dict.meta_data.get('input_dtype', jnp.float32))],
       config=config, rngs=init_rng)

  rng, init_rng = jax.random.split(rng)

  #Load model
  
  
  train_state = train_utils.TrainState(
    params=params, 
    model_state=model_state
  )
  chrono = train_utils.Chrono()
  if config.checkpoint:
    train_state, start_step = utils.restore_checkpoint(workdir, train_state)
  chrono.load(train_state.metadata['chrono'])
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

  knn_evaluator = knn_utils.KNNEvaluator(
    config,
    representation_fn_knn,
    knn_eval_batch_size,
    config.get("extract_only_descrs",False),
  )


  train_dir = config.get('train_dir')

  if config.test_pretrained_features:

    knn_utils.knn_step(
      knn_evaluator,
      train_state,
      config,
      train_dir,
      0,
      writer,
      config.preextracted,
    )

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

    knn_utils.knn_step(
      knn_evaluator,
      train_state,
      config,
      train_dir,
      step,
      writer,
      config.preextracted,
    )

  train_utils.barrier_across_hosts()



if __name__ == '__main__':
  app.run(main=knn_evaluate)