# pylint: disable=line-too-long
"""Default config for Dino training on ImageNet2012 for 100 epochs."""

import ml_collections

VARIANT = 'B/16'
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

  # Training.
  config.max_grad_norm = 1
  config.num_training_epochs = 100
  config.batch_size = 1024
  config.steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  config.rng_seed = 42
  total_steps = config.num_training_epochs * config.steps_per_epoch
  config.global_crops_scale = (0.4, 1.0) 
  config.local_crops_number = 0 #if 0, global scale = 0.14,1.0
  config.local_crops_scale = (0.05,0.4)
  config.student_temp = 0.1
  config.center_momentum = 0.9
  config.ncrops = 2
  config.warmup_teacher_temp = 0.04
  config.teacher_temp = 0.04
  config.warmup_teacher_temp_epochs = 0
  config.dataset_configs.number_of_focal_queries = n_queries - 1

  config.dataset_configs.pp_train = (
      'decode' +
      '|copy("image", "x1")' +
      '|copy("image", "x2")' +
      f'|copy_resize_file(224, {config.global_crops_scale}, inkey=("image", "x1"), outkey=("image", "x1"))' +
      '|copy_resize_file(224, inkey=("image", "x2"), outkey=("image", "x2"))' +
      '|value_range(0, 1, data_key="x1")' +
      '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x1")' +
      '|random_grayscale(0.2, data_key="x1")' +
      '|random_blur(1.0, data_key="x1")' +
      #f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x1")'
      '|value_range(0, 1, data_key="x2")' +
      '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x2")' +
      '|random_grayscale(0.2, data_key="x2")' +
      '|random_blur(1.0, data_key="x2")' +
      #f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x2")' +

      ''.join([f'|copy("image", "crops{i}")' for i in range(config.ncrops)]) +
      ''.join([f'|generate_crops((96, 96), {config.local_crops_scale}, inkey=("crops{i}"), outkey=("crops{i}"))' for i in range(config.ncrops)]) +
      #''.join([f'|flip(2, data_key="crops{i}")' for i in range(config.ncrops)]) + 
      ''.join([f'|value_range(0, 1, data_key="crops{i}")' for i in range(config.ncrops)]) +
      ''.join([f'|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="crops{i}")' for i in range(config.ncrops)]) +
      ''.join([f'|random_grayscale(0.2, data_key="crops{i}")' for i in range(config.ncrops)]) +
      ''.join([f'|random_blur(0.5, data_key="crops{i}")' for i in range(config.ncrops)]))
      #''.join([f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="crops{i}")' for i in range(config.ncrops)]) +
      #'|keep("x1, x2"' + ''.join([f', "crops{i}"' for i in range(config.ncrops)]) + ')')

  #'|copy_resize_file("image", "x1")' +
  #'|copy_resize_file("image", "x2")' +
  ''' '|init_patch_matching_tracker(14, "target_mask")' +
  '|init_box_tracker("target_box")' +
  f'|cropflip_generatemask({reference_resolution}, 32, flip=False, inkey=("reference", "target_mask", "target_box"), outkey=("reference", "target_mask", "target_box"))' +
  '|value_range(0, 1, data_key="reference")' +
  '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="reference")' +
  '|random_grayscale(0.2, data_key="reference")' +
  '|random_blur(1.0, data_key="reference")' +
  f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="reference")' +
  '|copy("image", "x1")' +
  '|value_range(0, 1, data_key="x1")' +
  '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x1")' +
  '|random_grayscale(0.2, data_key="x1")' +
  '|random_blur(1.0, data_key="x1")' +
  f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x1")' +
  '|copy("image", "x2")' +
  '|value_range(0, 1, data_key="x2")' +
  '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="x2")' +
  '|random_grayscale(0.2, data_key="x2")' +
  '|random_blur(1.0, data_key="x2")' +
  f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="x2")' +

  ''.join([f'|copy("image", "query{i}")' for i in range(n_queries)]) +
  '|inception_crop_with_mask((224, 224), 32, 100, (14, 14), inkey=("query0", "target_mask", "target_box"), outkey=("query0", "query0_mask", "query0_box"))' +
  ''.join([f'|inception_crop_with_mask((96, 96), 5, 32, (6, 6), inkey=("query{i}", "target_mask", "target_box"), outkey=("query{i}", "query{i}_mask", "query{i}_box"))' for i in range(1, n_queries)]) +
  ''.join([f'|flip_with_mask(inkey=("query{i}", "query{i}_mask"), outkey=("query{i}", "query{i}_mask"))' for i in range(n_queries)]) +
  ''.join([f'|value_range(0, 1, data_key="query{i}")' for i in range(n_queries)]) +
  ''.join([f'|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="query{i}")' for i in range(n_queries)]) +
  ''.join([f'|random_grayscale(0.2, data_key="query{i}")' for i in range(n_queries)]) +
  ''.join([f'|random_blur(0.5, data_key="query{i}")' for i in range(1, n_queries)]) +
  '|random_blur(0.1, data_key="query0")|random_solarize(0.2, data_key="query0")' +
  ''.join([f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="query{i}")' for i in range(n_queries)]) +
  '|keep("reference"' + ''.join([f', "query{i}", "query{i}_box", "query{i}_mask"' for i in range(n_queries)]) + ')')
  '''
  
  # For IMAGENET-1K
  #config.dataset_configs.dataset = 'imagenet2012'
  config.dataset_configs.dataset = 'imagenette'
  config.dataset_configs.train_split = 'train'

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
  config.model.head_output_dim = 4096
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.0
  config.model.stochastic_depth = 0.1
  config.model_dtype_str = 'float32'
  config.model.temperature = 0.1
  config.sharpening = 0.05

  # LOCA specific parameters.
  config.n_ref_positions = int((reference_resolution // patch)**2)
  config.apply_cluster_loss = True
  config.reference_seqlen = -1#int(0.2 * config.n_ref_positions)  # 20% of 196 is 39
  config.reference_seqlen_selection = 'consecutive'  # or 'unstructured' or 'first'
  config.query_max_seqlen = 70

  # Training.
  config.max_grad_norm = 1
  config.num_training_epochs = 800
  config.batch_size = 1024
  config.steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  config.rng_seed = 42
  total_steps = config.num_training_epochs * config.steps_per_epoch
  config.global_crops_scale = (0.4, 1.0) 
  config.local_crops_number = 0 #if 0, global scale = 0.14,1.0
  config.local_crops_scale = (0.05,0.4)
  config.student_temp = 0.1
  config.center_momentum = 0.9
  config.ncrops = 2
  config.warmup_teacher_temp = 0.04
  config.teacher_temp = 0.04
  config.warmup_teacher_temp_epochs = 0

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
  config.weight_decay = 0.1

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
  config.checkpoint_steps = 500
  config.log_summary_steps = 100

  return config


