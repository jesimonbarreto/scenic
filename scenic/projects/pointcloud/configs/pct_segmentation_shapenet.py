r"""Default configurations.

"""

import ml_collections

_TRAIN_SIZE = 12137
NUM_CLASSES = 50


def get_config(runlocal=''):
  """Get the default hyperparameter configuration."""
  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()

  config.experiment_name = 'pct-shapenet'
  config.log_summary_steps = 1 if runlocal else None

  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'shapenet.1.0.0'
  config.dataset_name = 'shapenet'
  config.data_dtype_str = 'float32'

  # Dataset config
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = 'shapenet'
  config.dataset_configs.num_classes = NUM_CLASSES
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.val_split = 'validation'
  config.dataset_configs.test_split = 'test'
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.onehot_labels = False

  # class rebalancing
  # config.class_rebalancing_factor = 1

  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000

  # Model details
  config.in_dim = 3
  config.feature_dim = 128
  config.max_seq_len = 2048
  config.kernel_size = 1
  config.dropout_rate = 0.5
  config.cache = False
  config.half_precision = False

  # Attention function configs
  config.use_attention_masking = False
  config.attention_masking_configs = ml_collections.ConfigDict()
  config.attention_masking_configs.nearest_neighbour_count = 256
  config.attention_masking_configs.use_knn_mask = True
  config.attention_masking_configs.mask_function = 'linear'

  config.attention_fn_cls = 'softmax'
  config.attention_fn_configs = ml_collections.ConfigDict()
  # config.attention_fn_configs.nonnegative_features = 'favorplusplus'
  config.attention_fn_configs.nb_features = 256
  config.attention_fn_configs.attention_kind = 'regular'

  # Training.
  config.trainer_name = 'segmentation_trainer'
  ## optimizer
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.01
  config.optimizer_configs.momentum = 0.9  # use for SGD

  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = 10000
  config.log_eval_steps = 1 if runlocal else 500
  config.batch_size = 2 if runlocal else 1024
  config.rng_seed = 42

  # Learning rate.
  steps_per_epoch = _TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 0.00001
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = base_lr

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  config.m = None  # Placeholder for randaug strength.
  config.l = None  # Placeholder for randaug layers.


  return config




