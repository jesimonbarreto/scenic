import ml_collections, os


VARIANT = 'B/16'
_IMAGENET_TRAIN_SIZE = 50000 #9469 #1281167
_IMAGENET_TEST_SIZE = 10000
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

def get_config():

  """Returns the ViT experiment configuration."""


  config = ml_collections.ConfigDict()
  config.experiment_name = '100ep_run'
  # Dataset.
  config.dataset_name = 'eval_dataset'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 250_000
  # For IMAGENET-1K
  #config.dataset_configs.dataset = 'imagenet2012'
  #for cifar 10
  config.dataset_configs.dataset = 'cifar10'
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.test_split = 'test'
  config.dataset_configs.batch_size_train = 1024
  config.dataset_configs.batch_size_test = 1024

  config.dataset_configs.pp_train = (
      'decode' +
      '|copy("image", "image_resized")' +
      '|onehot(10, key="label", key_result="label_onehot")' +
      f'|copy_resize_file(224, inkey=("image", "image_resized"), outkey=("image", "image_resized"))' +
      '|value_range(0, 1, data_key="image_resized")' +
      '|value_range(0, 1, data_key="image")' +
      f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="image_resized")' +
      f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="image")'
  )

  reference_resolution = 224
  n_queries = 10


  config.data_dtype_str = 'float32'
  #config.data_dtype_str = 'bfloat16'

  config.eval_batch_size = 32 #batch size for extracting embeddings
  config.knn_eval_batch_size = 32 #batch size for batch knn search 
  
  # Training.
  config.max_grad_norm = 1
  config.num_training_epochs = 400
  config.batch_size = 512
  config.steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  config.rng_seed = 42
  total_steps = config.num_training_epochs * config.steps_per_epoch
  config.global_crops_scale = (0.4, 1.0) 
  config.local_crops_number = 0 #if 0, global scale = 0.14,1.0
  config.local_crops_scale = (0.05,0.4)
  config.student_temp = 0.1
  config.center_momentum = 0.9
  config.ncrops = 8
  config.warmup_teacher_temp = 0.04
  config.teacher_temp = 0.04
  config.warmup_teacher_temp_epochs = 0
  config.dataset_configs.number_of_focal_queries = n_queries - 1

  
  #MODEL

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
  #LOCA

  # LOCA specific parameters.
  config.n_ref_positions = int((reference_resolution // patch)**2)
  config.apply_cluster_loss = True
  config.reference_seqlen = -1#int(0.2 * config.n_ref_positions)  # 20% of 196 is 39
  config.reference_seqlen_selection = 'consecutive'  # or 'unstructured' or 'first'
  config.query_max_seqlen = 70

  config.checkpoint = '/home/jesimonbarreto/scenic/checkpoint_501'

  ### kNN

  #dir of checkpoints
  config.train_dir = '/home/jesimonbarreto/scenic/dino_dir_test1/'

  config.preextracted = False
  #config.preextracted = True

  config.write_summary = True
  
  config.knn_start_step = 501
  config.knn_end_step = 1800 #set this to a lower value than start_epoch to not do knn at all
  config.knn_pass_step = 1299

  return config