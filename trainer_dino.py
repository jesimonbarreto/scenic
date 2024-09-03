"""DINO Training Script."""

import copy
import functools
from typing import Any, Callable, Dict, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform
from flax import jax_utils
import flax.linen as nn
from flax import traverse_util
from flax.core import freeze, unfreeze
from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state as ts
import jax
from jax import nn as opr
from jax.example_libraries import optimizers
import jax.numpy as jnp
import jax.profiler
import ml_collections
import optax
from scenic.dataset_lib import dataset_utils
import utils_dino as utils
import vit_dino as vit
from scenic.train_lib import lr_schedules
from scenic.train_lib import train_utils
import math, sys, os
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import wandb


# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]

def plot_example(train_batch, number_plot=5, dir_plot='/home/jesimonbarreto/images/', number_crops=2):
  
  def normalize_vector(vector):
    """Normalizes a JAX NumPy vector to values between 0 and 1."""
    min_val = jnp.min(vector)
    max_val = jnp.max(vector)
    return (vector - min_val) / (max_val - min_val)
  
  for stepe in range(number_plot):
    img = train_batch['x1'][0,stepe]
    print(f'1 max {jnp.max(img)} min {jnp.min(img)}')
    img = normalize_vector(img)
    print(f'2 max {jnp.max(img)} min {jnp.min(img)}')
    plt.imsave(os.path.join(dir_plot,f'imagex1_{stepe}.jpg'), img)  # Using matplotlib
    print(f'3 max {jnp.max(img)} min {jnp.min(img)}')
    img = train_batch['x2'][0,stepe]
    print(f'4 max {jnp.max(img)} min {jnp.min(img)}')
    img = normalize_vector(img)
    plt.imsave(os.path.join(dir_plot,f'imagex2_{stepe}.jpg'), img)
    for vcrop in range(number_crops):
      print(f'{vcrop} de {number_crops}')
      img = train_batch[f'crops{vcrop}'][0,stepe]
      img = normalize_vector(img)
      plt.imsave(os.path.join(dir_plot,f'crops{vcrop}_{stepe}.jpg'), img)

def dino_train_step(
    train_state: utils.TrainState,
    batch: Batch,
    center: jnp.ndarray,
    epoch: int,
    *,
    flax_model: nn.Module,
    momentum_parameter_scheduler: Callable[[int], float],
    loss_fn: Any,
    metrics_fn: Any,
    steps_per_epoch: float,
    config: ml_collections.ConfigDict,
) -> Tuple[utils.TrainState, Dict[str, Tuple[float, int]]]:
  """Runs a single step of training.

  Given the state of the training and a batch of data, computes
  the loss and updates the parameters of the model.

  Args:
    train_state: The state of training including the current
      global_step, model_state, rng, and optimizer. The buffer of this argument
      can be donated to the computation.
    batch: A single batch of data. The buffer of this argument can be donated to
      the computation.
    flax_model: A Flax model.
    momentum_parameter_scheduler: Momentum parameter scheduler for EMA update.
    loss_fn: The cross-entropy loss function.
    metrics_fn: Reports relative position loss and accuracy.
    config: Configurations of the experiment.

  Returns:
    The updated state of training.
  """
  # Some preparations.
  new_rng, dropout_rng, droptok_rng = jax.random.split(train_state.rng, num=3)
  dropout_rng = train_utils.bind_rng_to_host_device(
      dropout_rng, axis_name='batch', bind_to='device')
  droptok_rng = train_utils.bind_rng_to_host_device(
      droptok_rng, axis_name='batch', bind_to='device')
  step = train_state.global_step
  momentum_parameter = momentum_parameter_scheduler(step)
  n_pos = config.n_ref_positions  # Number of reference positions.
  bs = batch['x1'].shape[0]  # Per-device batch size.
  n_q_foc = config.dataset_configs.number_of_focal_queries
  batch = utils.prepare_input(batch, config)

  def training_loss_fn(params, center, epoch):
    # Step 1): Predict teacher network, predict student.
    # get features
    use_ema = config.apply_cluster_loss
    drop_moment = 'late' if config.apply_cluster_loss else 'early'

    teacher_out = flax_model.apply(
        {'params': train_state.ema_params if use_ema else params},
        batch['sample'][0],
        seqlen=config.reference_seqlen,
        seqlen_selection=config.reference_seqlen_selection,
        drop_moment=drop_moment,
        backbone = True,
        train=True,
        rngs={'dropout': dropout_rng, 'droptok': droptok_rng})["x_train"]
    
    st = flax_model.apply(
        {'params': params},
        batch['sample'][0],
        seqlen=config.reference_seqlen,
        seqlen_selection=config.reference_seqlen_selection,
        drop_moment=drop_moment,
        backbone = True,
        train=True,
        rngs={'dropout': dropout_rng, 'droptok': droptok_rng})["x_train"]
    
    '''cc = flax_model.apply(
        {'params': params},
        batch['sample'][1],
        seqlen=config.reference_seqlen,
        seqlen_selection=config.reference_seqlen_selection,
        drop_moment=drop_moment,
        backbone = True,
        train=True,
        rngs={'dropout': dropout_rng, 'droptok': droptok_rng})'''
    
    #student_out = jnp.concatenate([st,cc])
    student_out = st
    loss_dino, center = loss_fn(teacher_out, student_out, center, epoch)
    total_loss = loss_dino
    if config.mode == 'random':
      teacher_out = flax_model.apply(
        {'params': train_state.ema_params if use_ema else params},
        batch['sample'][1],
        seqlen=config.reference_seqlen,
        seqlen_selection=config.reference_seqlen_selection,
        drop_moment=drop_moment,
        backbone = True,
        train=True,
        rngs={'dropout': dropout_rng, 'droptok': droptok_rng})["x_train"]
    
      st = flax_model.apply(
          {'params': params},
          batch['sample'][1],
          seqlen=config.reference_seqlen,
          seqlen_selection=config.reference_seqlen_selection,
          drop_moment=drop_moment,
          backbone = True,
          train=True,
          rngs={'dropout': dropout_rng, 'droptok': droptok_rng})["x_train"]
      student_out = st
      loss_dino, center = loss_fn(teacher_out, student_out, center, epoch)
      total_loss += loss_dino
      total_loss /=2
    return total_loss, (loss_dino, center)
  
  compute_gradient_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
  (total_loss, (loss_dino, center)), grad = compute_gradient_fn(
      train_state.params, center, epoch)
  #metrics = metrics_fn(logits, batch)
  metrics = (
      dict(total_loss=(total_loss, 1)))

  # Update the network parameters.
  grad = jax.lax.pmean(grad, axis_name='batch')
  if config.get('max_grad_norm', None) is not None:
    grad = optimizers.clip_grads(grad, config.max_grad_norm)
  new_train_state = train_state
  if train_state.tx is not None:
    updates, new_opt_state = train_state.tx.update(
        grad, train_state.opt_state, train_state.params)
    new_params = optax.apply_updates(train_state.params, updates)

    # update the teacher weights
    new_ema_params = jax.tree_map(
        lambda s, t: momentum_parameter * t + (1 - momentum_parameter) * s,
        new_params, train_state.ema_params)

    new_train_state = train_state.replace(  # pytype: disable=attribute-error
        global_step=step + 1,
        opt_state=new_opt_state,
        params=new_params,
        ema_params=new_ema_params,
        rng=new_rng)
  return new_train_state, center, metrics


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[Any, Any]:
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the utils.TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    dataset: The dataset that has train_iter and meta_data.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_state that has the state of training.
  """


  lead_host = jax.process_index() == 0
  #plot flag
  fstexe = True

  # Build the loss_fn, metrics, and flax_model.
  model = vit.ViTDinoModel(config, dataset.meta_data)

  # Start a run, tracking hyperparameters
  wandb.init(
      # set the wandb project where this run will be logged
      project=config.project,
      name=config.experiment_name,
      # track hyperparameters and run metadata with wandb.config
      config=dict(config)
  )


  num_local_devices = jax.local_device_count()

  #Center used to calculate the loss
  center = jnp.zeros((num_local_devices, config.model.head_output_dim))
  
  # Randomly initialize model parameters.
  rng, init_rng = jax.random.split(rng)
  (params, _, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config, rngs=init_rng)
  
  '''  # Função para listar todas as camadas
  def list_layers(params, parent_name=""):
      layer_names = []
      for layer_name, layer_params in params.items():
          full_name = f"{parent_name}/{layer_name}" if parent_name else layer_name
          if isinstance(layer_params, dict):
              # Recursivamente lista subcamadas
              layer_names.extend(list_layers(layer_params, full_name))
          else:
              # Adiciona o nome da camada atual
              layer_names.append(full_name)
      return layer_names

  # Lista os nomes de todas as camadas
  layer_names = list_layers(params)
  for name in layer_names:
      print(name)'''

  '''=============================================='''
  #print(f'Here... trying load {params.keys()}')
  from load_params import load_params

  params = load_params(config.load_weights,'/home/jesimonbarreto/', params,
                params_key='teacher_weights',
                force_random_init= None)


  #print(f'Here... finished load {params.keys()}')
  '''=============================================='''

  # Only one model function but two sets of parameters.
  ema_params = copy.deepcopy(params)

  # Get learning rate and ema temperature schedulers.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)
  momentum_parameter_scheduler = lr_schedules.compound_lr_scheduler(
      config.momentum_rate)

  #weight_decay_mask = jax.tree_map(lambda x: x.ndim != 1, params)
  # Create optimizer.
  if config.transfer_learning:
    params = freeze(params)
    
    def create_mask(params, label_fn):
      def _map(params, mask, label_fn):
          for k in params:
              if label_fn(k):
                  mask[k] = 'zero'
              else:
                  if isinstance(params[k], FrozenDict):
                      mask[k] = {}
                      _map(params[k], mask[k], label_fn)
                  else:
                      mask[k] = 'adam'
      mask = {}
      _map(params, mask, label_fn)
      return frozen_dict.freeze(mask)

    def zero_grads():
        # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
        def init_fn(_):
            return ()
        def update_fn(updates, state, params=None):
            return jax.tree_map(jnp.zeros_like, updates), ()
        return optax.GradientTransformation(init_fn, update_fn)

    tx = optax.multi_transform(
        {'adam': optax.inject_hyperparams(optax.adamw)(
        learning_rate=learning_rate_fn, weight_decay=config.weight_decay,),
        'zero': zero_grads()},
         create_mask(params, lambda s: 'encoder' in s or 'ToTokenSequence' in s)
        )
    
    print(create_mask(params, lambda s: 'encoder' in s or 'ToTokenSequence' in s))
    
  else:
    tx = optax.inject_hyperparams(optax.adamw)(
        learning_rate=learning_rate_fn, weight_decay=config.weight_decay,
        )#mask=weight_decay_mask,)
    
  opt_state = jax.jit(tx.init, backend='cpu')(params)

  # Create chrono class to track and store training statistics and metadata.
  chrono = train_utils.Chrono()

  
  # Create the TrainState to track training state (i.e. params and optimizer).
  train_state = utils.TrainState(
    global_step=0, opt_state=opt_state, tx=tx, params=params,
      ema_params=ema_params, rng=rng, metadata={'chrono': chrono.save()})
  
  if config.save_state_0:
    unrep_train_state = train_state
    metadata = unrep_train_state.metadata
    metadata['chrono'] = chrono.save()
    unrep_train_state.replace(metadata=metadata)  # pytype: disable=attribute-error
    utils.save_checkpoint(workdir, unrep_train_state)
    
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = utils.restore_checkpoint(workdir, train_state)

  chrono.load(train_state.metadata['chrono'])
  train_state = train_state.replace(metadata={})
  # Replicate the training state: optimizer, params and rng.
  train_state = jax_utils.replicate(train_state)
  del params, ema_params
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)

  # The function that performs one step of loca training.
  dino_train_step_pmapped = jax.pmap(
      functools.partial(
          dino_train_step,
          flax_model=model.flax_model,
          loss_fn=model.loss_function,
          metrics_fn=model.get_metrics_fn,
          momentum_parameter_scheduler=momentum_parameter_scheduler,
          steps_per_epoch = steps_per_epoch,
          config=config),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0,1),
  )

  train_metrics, train_summary, ext_log = [], None, []
  chrono.inform(start_step, total_steps, config.batch_size, steps_per_epoch)
  report_progress = periodic_actions.ReportProgress(num_train_steps=total_steps,
                                                    writer=writer)
  def write_note(note):
    if lead_host:
      platform.work_unit().set_notes(note)
  hooks = []
  if lead_host:
    hooks.append(report_progress)
  if config.get('xprof', True) and lead_host:
    hooks.append(periodic_actions.Profile(num_profile_steps=5, logdir=workdir))
  if start_step == 0:
    step0_log = {'num_trainable_params': num_trainable_params}
    if gflops:
      step0_log['gflops'] = gflops
    writer.write_scalars(1, step0_log)
  logging.info('Starting training loop at step %d.', start_step + 1)
  v={}
  for step in range(start_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train', step_num=step):
      epoch = jnp.ones((num_local_devices, 1))*step/steps_per_epoch
      epoch = epoch.astype(jnp.int32)
      train_batch = next(dataset.train_iter)

      if config.plot_ex and fstexe:
        print(f' config {config.ncrops}')
        plot_example(train_batch, 
                     number_plot=config.number_plot,
                     dir_plot=config.dir_plot,
                     number_crops=config.ncrops)
        fstexe = False

      train_state, center, tm = dino_train_step_pmapped(
                                  train_state,
                                  train_batch,
                                  center,
                                  epoch)
      v['learning_rate'] = train_state.opt_state.hyperparams['learning_rate']
      ext_log.append(v)
      train_metrics.append(tm)
    for h in hooks:
      h(step)
    ###################### LOG TRAIN SUMMARY ########################
    if (step % config.get('log_summary_steps') == 1) or (step == total_steps):
      chrono.pause()
      if lead_host:
        chrono.tick(step, writer, write_note)
      train_summary = train_utils.log_train_summary(
          step=step,
          train_metrics=jax.tree_util.tree_map(train_utils.unreplicate_and_get,
                                               train_metrics),
          extra_training_logs= ext_log,
          writer=writer)
      print(ext_log)
      wb = train_utils.stack_forest(ext_log)
      for key, val in wb.items():
        train_summary[key]=float(val.mean())
      print(train_summary)
      wandb.log(train_summary, step=step)
      chrono.resume()
      train_metrics = []
      ext_log = []
    ##################### CHECKPOINTING ###################
    if ((step % config.get('checkpoint_steps') == 1 and step > 1) or
        (step == total_steps)) and config.checkpoint:
      chrono.pause(wait_for=(train_state.params, train_state.opt_state))
      with report_progress.timed('checkpoint'):
        if lead_host:
          # Take the first replica.
          unrep_train_state = jax_utils.unreplicate(train_state)
          metadata = unrep_train_state.metadata
          metadata['chrono'] = chrono.save()
          unrep_train_state.replace(metadata=metadata)  # pytype: disable=attribute-error
          utils.save_checkpoint(workdir, unrep_train_state, max_to_keep=config.max_keep_checkpoint)
          del unrep_train_state
      chrono.resume()  # Un-pause now.
  # Wait until computations are done before exiting.
  train_utils.barrier_across_hosts()
  # Return the train summary after last step.
  return train_state, train_summary
