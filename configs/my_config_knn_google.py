import ml_collections, os
import jax.numpy as jnp
VARIANT = 'S/14'
_IMAGENET_TRAIN_SIZE = 1281167 #9469 #1281167
_IMAGENET_TEST_SIZE = 50000
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]
MEAN = [0.5]
STD = [0.5]

def get_config():
  global _IMAGENET_TRAIN_SIZE, _IMAGENET_TEST_SIZE
  """Returns the ViT experiment configuration."""
  config_val = ml_collections.ConfigDict()
  #WANDB
  config_val.project = 'Eval_report'
  config_val.experiment_name = 'Eval_Dino_8k2Mhead_all'
  config_val.extract_train = True
  # Dataset.
  config_val.dataset_name = 'eval_dataset'
  config_val.data_dtype_str = 'float32'
  config_val.dataset_configs = ml_collections.ConfigDict()
  config_val.dataset_configs.prefetch_to_device = 2
  config_val.dataset_configs.shuffle_buffer_size = 250_000
  # For IMAGENET-1K
  #config_val.dataset_configs.dataset = 'imagenet2012'
  #for cifar 10
  config_val.dataset_configs.dataset = 'imagenet2012'
  config_val.dataset_configs.dataset_dir = '/mnt/disks/dataset/dataset/imagenet/'
  config_val.dataset_configs.train_split = 'train'
  config_val.dataset_configs.test_split = 'validation'
  config_val.dataset_configs.batch_size_train = 256
  config_val.dataset_configs.batch_size_test = 64
  config_val.num_classes = 1000
  reference_resolution = 224
  crop_size = 224
  config_val.T = 0.07

  config_val.dataset_configs.filter_classes = True
  if config_val.dataset_configs.filter_classes:
    config_val.dataset_configs.desired_classes = [
                                              897, 827, 764, 761, 742, 721, 651,
                                              650, 637, 632, 620, 738, 534, 508,
                                              435, 412, 879, 859, 463, 470, 481,
                                              473, 587, 313, 872, 629, 745, 760,
                                              963, 938, 937, 987, 943, 955, 953,
                                              957, 954, 752, 792, 626, 951, 112,
                                              928
                                              ]
    #update number classes variables
    config_val.num_classes_filter = config_val.num_classes#len(config_val.dataset_configs.desired_classes)
    _IMAGENET_TRAIN_SIZE = 1281167#732-1300 per class in the ILSVRC2012 training set. #update quantity samples train each class selected
    _IMAGENET_TEST_SIZE = 2134#update quantity samples train each class selected
  else:
    config_val.num_classes_filter = config_val.num_classes


  config_val.dataset_configs.pp_train = (
      'decode' +
      '|copy("image", "image_resized")' +
      #f'|adjust_labels({config_val.dataset_configs.desired_classes}, {config_val.num_classes},{config_val.dataset_configs.filter_classes}, key="label", key_result="label_adj")' +
      f'|onehot({config_val.num_classes_filter}, key="label", key_result="label_onehot")' +
      '|resize_small(256, data_key="image")'+
      '|resize_small(256, data_key="image_resized")'+
      '|central_crop(224, data_key="image")'+
      '|central_crop(224, data_key="image_resized")'+
      '|value_range(0, 1, data_key="image")' +
      '|value_range(0, 1, data_key="image_resized")' +
      f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="image")'+
      f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="image_resized")'+
      '|keep("image", "image_resized", "label", "label_onehot")'
      #'|keep("image", "image_resized", "label_adj", "label", "label_onehot")'
  )


  ### kNN

  #dir of checkpoints
  config_val.train_dir = '/home/jesimonbarreto/weights/'#'/home/jesimonbarreto/exp_test_now/'
  config_val.preextracted = False
  config_val.write_summary = True
  #finetun_ckp_10778- 1  layerwise_ckp_10778-2  lr00001better_ckp_10778-3  lr0001_ckp_10778-4
  config_val.steps_checkpoints = [1,2,3,4]
  config_val.ks = [5,10,20]
  config_val.dir_files = '/mnt/disks/dataset/eval_files/'

  config_val.data_dtype_str = 'float32'
  #config_val.data_dtype_str = 'bfloat16'


  
  config_val.batch_size = config_val.dataset_configs.batch_size_train #batch size for extracting embeddings
  #config_val.knn_eval_batch_size = 32 #batch size for batch knn search 

  config_val.checkpoint = False

  return config_val