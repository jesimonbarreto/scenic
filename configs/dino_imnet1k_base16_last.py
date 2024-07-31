# pylint: disable=line-too-long
"""Default config for Dino training on ImageNet2012 for 100 epochs."""

import ml_collections

VARIANT = 'B/14'
_IMAGENET_TRAIN_SIZE = 9469 #1281167
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def get_config():
  """Returns the default config for a 100 epoch DINO training on ImageNet2012."""

  config = ml_collections.ConfigDict()
  config.experiment_name = '100ep_run'
  # Dataset.
  config.dataset_name = 'dino_dataset'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 250_000
  reference_resolution = 224
  n_queries = 10
  config.mode = 'video'
  
  #plot
  config.plot_ex = False
  config.number_plot = 5
  config.dir_plot = '/home/jesimonbarreto/images/'

  # Training.'MVImagenet'
  config.max_grad_norm = 1
  config.num_training_epochs = 2#400
  config.batch_size = 512
  config.steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  config.rng_seed = 42
  total_steps = config.num_training_epochs * config.steps_per_epoch
  config.global_crops_scale = (0.4, 1.0) 
  config.local_crops_number = 0 #if 0, global scale = 0.14,1.0
  config.local_crops_scale = (0.05,0.4)
  config.student_temp = 0.1
  config.center_momentum = 0.9
  config.ncrops = 0 #change other parameters
  config.warmup_teacher_temp = 0.04
  config.teacher_temp = 0.04
  config.warmup_teacher_temp_epochs = 0
  config.dataset_configs.number_of_focal_queries = n_queries - 1

  config.dataset_configs.pp_train = (
      'decode' +
      f'|copy_video("video", mode={config.mode})' +
      f'|copy_resize_file(224, {config.global_crops_scale}, inkey=("x1", "x1"), outkey=("x1", "image"))' +
      f'|copy_resize_file(224, {config.global_crops_scale}, inkey=("x2", "x2"), outkey=("x2", "image"))' +
      '|value_range(0, 1, data_key="x1")' +
      '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x1")' +
      '|random_grayscale(0.2, data_key="x1")' +
      '|random_blur(1.0, data_key="x1")' +
      f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x1")'

      '|value_range(0, 1, data_key="x2")' +
      '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x2")' +
      '|random_grayscale(0.2, data_key="x2")' +
      '|random_blur(0.1, data_key="x2")' +
      '|random_solarize(0.2, data_key="x2")' +
      f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x2")')
  if config.mode == 'random':
    config.dataset_configs.pp_train += (
      f'|copy_resize_file(224, {config.global_crops_scale}, inkey=("x3", "x3"), outkey=("x3", "image"))' +
      f'|copy_resize_file(224, {config.global_crops_scale}, inkey=("x4", "x4"), outkey=("x4", "image"))' +
      '|value_range(0, 1, data_key="x3")' +
      '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x3")' +
      '|random_grayscale(0.2, data_key="x3")' +
      '|random_blur(1.0, data_key="x3")' +
      f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x3")'

      '|value_range(0, 1, data_key="x4")' +
      '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x4")' +
      '|random_grayscale(0.2, data_key="x4")' +
      '|random_blur(0.1, data_key="x4")' +
      '|random_solarize(0.2, data_key="x4")' +
      f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x4")' +
      '|keep("video, x1, x2, x3, x4, label")'
      )
  else:
    config.dataset_configs.pp_train += (
      '|keep("video, x1, x2, label")'
    )
  
  # For IMAGENET-1K
  #config.dataset_configs.dataset = 'imagenet2012'
  config.dataset_configs.dataset = 'mvimgnet'
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.dataset_dir = '/mnt/disks/persist/dataset/imagenet/'


  # Model.
  version, patch = VARIANT.split('/')
  patch = int(patch)
  config.model = ml_collections.ConfigDict()
  config.model.hidden_size = {'Ti': 192,
                              'S': 384,
                              'B': 768,
                              'L': 1024,
                              'H': 1280}[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [patch, patch]
  config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}[version]
  config.model.mlp_dim = {'Ti': 768,
                          'S': 1536,
                          'B': 3072,
                          'L': 4096,
                          'H': 5120}[version]
  config.model.num_layers = {'Ti': 12,
                             'S': 12,
                             'B': 12,
                             'L': 24,
                             'H': 32}[version]
  config.model.head_output_dim = 4096 #65536 
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.0
  config.model.stochastic_depth = 0.1
  config.model_dtype_str = 'float32'
  config.model.temperature = 0.1
  config.sharpening = 0.05
  #Verificar esses fatores no codigo
  config.norm_last_layer = True
  config.momentum_teacher = 0.996
  config.use_bn_in_head = False

  # LOCA specific parameters.
  config.n_ref_positions = int((reference_resolution // patch)**2)
  config.apply_cluster_loss = True
  config.reference_seqlen = -1#int(0.2 * config.n_ref_positions)  # 20% of 196 is 39
  config.reference_seqlen_selection = 'consecutive'  # or 'unstructured' or 'first'
  config.query_max_seqlen = 70

  # Learning rate.
  #cosine schedule lr
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = config.steps_per_epoch * 15
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = 0.001 * config.batch_size / 1024
  config.lr_configs.alpha = 0.01

  # Weight decay.
  config.weight_decay = 0.04
  #verificar
  config.weight_decay_end = 0.4
  config.lr=0.0005
  config.warmup_epochs=10
  config.optimizer = 'adamw'
  config.drop_path_rate= 0.1

  # Momentum rate scheduler.
  config.momentum_rate = ml_collections.ConfigDict()
  config.momentum_rate.factors = 'constant*cosine_decay'
  config.momentum_rate.steps_per_cycle = total_steps
  config.momentum_rate.base_learning_rate = 0.996
  config.momentum_rate.alpha = 1. / 0.996

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 50
  config.log_summary_steps = 5

  return config


