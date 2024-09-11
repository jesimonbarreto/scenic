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

import tensorflow as tf
from PIL import Image
import numpy as np

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
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

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

  embedding = flax_model.apply(
        {'params': train_state.params},
        batch,
        seqlen=-1,
        seqlen_selection='consecutive',
        drop_moment='late',
        backbone = True,
        train=False)
  
  #embedding = normalize(embedding)

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
  #dataset = train_utils.get_dataset(
  #    config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  
  train(
      rng=rng,
      config=config,
      dataset=None,
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
  meta_data = {
      'input_shape': (1, 224, 224, 3),
      'num_train_examples': 1,
      'input_dtype': getattr(jnp, 'float32'),
      'label_data' : 1,
  }
  
  model = vit.ViTDinoModel(config,meta_data)

  # Randomly initialize model parameters.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(meta_data['input_shape'],
                    meta_data.get('input_dtype', jnp.float32))],
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
  train_dir= '/home/jesimonbarreto/test/'
  step = 10778
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
    '''=============================================='''
    print('Here... trying load')
    from load_params import load_params
    print(f' {config.dir_weight} {config.weight_load}')
    params = load_params(config.weight_load,config.dir_weight, params,
                  params_key='teacher_weights',
                  force_random_init= None)

    print('Here... finished load')
    '''=============================================='''
    # Only one model function but two sets of parameters.
    ema_params = copy.deepcopy(params)
    # Create the TrainState to track training state (i.e. params and optimizer).
    train_state = utils.TrainState(
        global_step=0, opt_state=opt_state, tx=tx, params=params,
        ema_params=ema_params, rng=rng, metadata={'chrono': chrono.save()})
    train_state = jax_utils.replicate(train_state)
      
  #project feats or not
  representation_fn_knn = functools.partial(
      representation_fn_eval,
      flax_model = model.flax_model, 
  )
  repr_fn = jax.pmap(
          representation_fn_knn, 
          #donate_argnums=(1,),
          axis_name='batch',
  )

    # extract features
  @jax.jit
  def extract_features(batch):
    features = repr_fn(train_state, batch)
    return features  # Return extracted features for the batch
  
  def to_tensor(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0
    # Transpose to (C x H x W) format
    #image = tf.transpose(image, perm=[0, 1, 2])
    return image
  
  def center_crop(image, crop_size):
    pad_width = 2
    # Define the padding configuration
    padding = [[pad_width, pad_width], [pad_width, pad_width], [0, 0]]
    image = tf.pad(image, padding, mode='CONSTANT', constant_values=0)
    
    # Get the dimensions of the image
    height, width = image.shape[0], image.shape[1]

    # Calculate the crop offsets
    offset_height = int((height - crop_size[0]) / 2)
    offset_width = int((width - crop_size[1]) / 2)

    # Crop the image
    cropped_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_size[0], crop_size[1])
    return cropped_image
  
  def resize(image, target_size):
    # Get the dimensions of the image
    height, width = image.shape[0], image.shape[1]

    # Determine the scaling factor to make the smaller dimension 224
    scale_factor = tf.cond(height < width,
                          lambda: target_size / height,
                          lambda: target_size / width)

    # Resize the image while preserving the aspect ratio
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    resized_image = tf.image.resize(image, [new_height, new_width])

    return resized_image
  
  # Define image transformation pipeline using tf.image
  def transform_image(image):
      image = tf.convert_to_tensor(image, dtype=tf.float32)
      image = resize(image, 224)  # Resize image
      image = center_crop(image, crop_size=(224,224))  # Center crop image
      image = to_tensor(image)  # Convert image to float32
      image = (image - [0.5]) / [0.5]  # Normalize using provided mean and std dev
      return image

  ##########################################################################################
    # Load image
  for name_img in glob.glob('/mnt/disks/dataset/mvimgnet/copy/*.*'):
    img = Image.open(name_img).convert('RGB')
    resul_name = name_img.split('/')[-1].split('.')[0]

    # Apply transformation to the image
    img = transform_image(img)
    #img_save = Image.fromarray((np.array((img*0.5)+0.5)*255).transpose(1, 2, 0).astype(np.uint8))
    #img_save.save('/home/jesimon/Documentos/mestrado/meta_dog_features_jax_input.png')

    #np.save('/home/jesimon/Documentos/mestrado/py_image_proce_jax.npy',img)

    # Add batch dimension
    #img = jnp.array(np.load('/home/jesimon/Documentos/mestrado/py_image_proce.npy'))
    img = tf.expand_dims(img, 0)
    img = tf.expand_dims(img, 0)

    ######################################################################################  
    img = jnp.array(img)
    img = jnp.tile(img, (8, 8, 1, 1, 1))
    print(img.shape)
    result = extract_features(img)
    #result = jnp.squeeze(result)
    img = jnp.squeeze(result['x_norm_patchtokens'][0, 0, 0])
    print(img.shape)
    
    img = np.array(img)

    pca = PCA(n_components=3)
    pca.fit(img)

    pca_features = pca.transform(img)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    plt.imshow(pca_features.reshape(16, 16, 3).astype(np.uint8))
    plt.savefig('/home/jesimonbarreto/gen_random/'+resul_name+'.png')

if __name__ == '__main__':
  app.run(main=knn_evaluate)
