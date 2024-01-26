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

  


  ### kNN

  #dir of checkpoints
  config.train_dir = '/home/jesimonbarreto/scenic/dino_dir_test1/'
  

  return config