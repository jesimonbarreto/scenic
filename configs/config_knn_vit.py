import ml_collections, os


DATASET_TRAIN_SIZE = { #number of train images per dataset
    'cars': 6346,
    'sop': 48942,
    'inshop': 20897,
    'inat': 273929,
    'met': 397121,
    'gldv2': 1422914,
    'food2k': 472349,
    'rp2k': 188724,
}


ViT_configs = {
    'B/16': {
    "hidden_size" : 768,
    "patches_size" : [16, 16],
    "num_heads" : 12,
    "mlp_dim" : 3072,
    "num_layers" : 12,
    "checkpoint" : 'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'
  },
}



def get_aggregated_size(datasets):

  size = 0
  
  for dataset in datasets.split(','):
    size += DATASET_TRAIN_SIZE[dataset]
  
  return size


def get_config():

  """Returns the ViT experiment configuration."""

  config = ml_collections.ConfigDict()

  config.experiment_name = 'universal-embedding-vit'

  # Dataset that was used for training.
  config.dataset_name = "imagenette,"
  config.knn_eval_names = "imagenette,"

  config.data_dtype_str = 'float32'
  #config.data_dtype_str = 'bfloat16'

  config.eval_batch_size = 1024 #batch size for extracting embeddings
  config.knn_eval_batch_size = 1024 #batch size for batch knn search 
  
  #merged eval
  config.disabled_separate_knns = 'train_knn,val_knn,test_knn'
  config.disabled_merged_knns = 'train_knn,val_knn'

  #separate eval
  # config.disabled_separate_knns = 'train_knn,val_knn'
  # config.disabled_merged_knns = 'train_knn,val_knn,test_knn'

  config.dataset_configs = ml_collections.ConfigDict()
  
  config.classifier = "joint" 
  #config.classifier = "separate"

  config.count_flops = False #bugged?

  # Model.
  config.model_class = 'vit_with_embedding'
  
  config.model_type = "B/16"

  config.clip = False

  config.model = ml_collections.ConfigDict()

  #move these to the dictionary of the models
  config.model.hidden_size = ViT_configs[config.model_type]["hidden_size"]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = ViT_configs[config.model_type]["patches_size"]
  config.model.num_heads = ViT_configs[config.model_type]["num_heads"]
  config.model.mlp_dim = ViT_configs[config.model_type]["mlp_dim"]
  config.model.num_layers = ViT_configs[config.model_type]["num_layers"]
  config.model.representation_size = None #we will always use that as None
  
  config.model.output_dim = 64 #our chosen embedding dimension
  
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.0
  
  config.model_dtype_str = 'float32'
  #config.model_dtype_str = 'bfloat16'
  
  #config.model.positional_embedding = 'none'
  config.model.positional_embedding = 'learned_1d'

  config.transform_logits_type = 'normface'

  #checkpoints
  config.pretrained_ckpt_dir = ''
  config.pretrained_ckpt = os.path.join(config.pretrained_ckpt_dir,ViT_configs[config.model_type]["checkpoint"])
  
  # Training.
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 1e-6
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  # config.max_grad_norm = 1.0
  config.label_smoothing = None

  config.num_training_epochs = 30
  
  config.batch_size = 128

  config.steps_per_epoch = (get_aggregated_size(config.dataset_name) // config.batch_size)

  config.rng_seed = 0 

  config.init_head_bias = -10.0 
  config.loss = ml_collections.ConfigDict()
  config.loss.m = 0.0
  config.loss.scale = 16
  config.max_to_keep = 1000

  config.log_eval_steps = config.steps_per_epoch

  # Learning rate.
  base_lr = 1e-3

  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant'
  config.lr_configs.base_learning_rate = base_lr

  config.lr_configs.backbone = ml_collections.ConfigDict()

  #that means for 2 epochs we train only the classifier
  config.lr_configs.backbone.frozen_steps = (
      2 * config.steps_per_epoch
  )

  config.lr_configs.backbone.base_learning_rate = base_lr * 1e-2

  # kNN

  #dir of checkpoints
  config.train_dir = ''
  
  # Logging.

  config.preextracted = False
  #config.preextracted = True

  config.write_summary = True
  
  config.test_pretrained_features = False

  config.extract_only_descrs = False

  config.checkpoint = False 

  config.save_descriptors = True
  #config.save_descriptors = False

  config.debug_eval = False  # Debug mode during eval.
    
  config.eval_dataset_dir = ''
  config.train_dataset_dir = '' 
  
  config.project_feats_knn = True
  #config.project_feats_knn = False

  config.top_k = 5 #top k neighbors to look at
  #config.top_k = 100

  config.knn_start_epoch = 3
  config.knn_end_epoch = 3 #set this to a lower value than start_epoch to not do knn at all

  config.log_csv = False

  config.save_neighbors = False

  config.info_files_dir = ''

  #config.descr_save_path = "."
  config.descr_save_path = None

  return config






#################Editado


# pylint: disable=line-too-long
"""Default config for LOCA training on ImageNet2012 for 100 epochs."""

import ml_collections
import os

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
  config.knn_eval_names = "food2k,cars,sop,inshop,inat,met,gldv2,rp2k"
  config.data_dtype_str = 'float32'

  config.eval_batch_size = 1024 #batch size for extracting embeddings
  config.knn_eval_batch_size = 1024 #batch size for batch knn search 

  #merged eval
  config.disabled_separate_knns = 'train_knn,val_knn,test_knn'
  config.disabled_merged_knns = 'train_knn,val_knn'
  
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.prefetch_to_device = 2
  config.dataset_configs.shuffle_buffer_size = 250_000
  reference_resolution = 224
  config.classifier = "joint" 
  #config.classifier = "separate"

  config.count_flops = False #bugged?

  # Model.
  config.model_class = 'vit_with_embedding'
  
  config.model_type = "B/16"

  config.clip = False
  
  config.dataset_configs.pp_train = (
      'decode' +
      '|copy("image", "reference")' +
      '|init_patch_matching_tracker(14, "target_mask")' +
      '|init_box_tracker("target_box")' +
      f'|cropflip_generatemask({reference_resolution}, 32, flip=False, inkey=("reference", "target_mask", "target_box"), outkey=("reference", "target_mask", "target_box"))' +
      '|value_range(0, 1, data_key="reference")' +
      '|random_color_jitter(0.8, 0.4, 0.4, 0.2, 0.1, data_key="reference")' +
      '|random_grayscale(0.2, data_key="reference")' +
      '|random_blur(1.0, data_key="reference")' +
      f'|standardize({MEAN_RGB}, {STDDEV_RGB}, data_key="reference")' +
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
      
  
  # For IMAGENET-1K
  #config.dataset_configs.dataset = 'imagenet2012'
  config.dataset_configs.dataset = 'imagenette'
  config.dataset_configs.train_split = 'validation'

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
  config.num_training_epochs = 100
  config.batch_size = 1024
  config.steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  config.rng_seed = 42
  total_steps = config.num_training_epochs * config.steps_per_epoch
  config.global_crops_scale = (0.4, 1.0) 
  config.local_crops_number = 8 #if 0, global scale = 0.14,1.0
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


  #config.model.positional_embedding = 'none'
  config.model.positional_embedding = 'learned_1d'

  config.transform_logits_type = 'normface'

  #checkpoints
  config.pretrained_ckpt_dir = ''
  config.pretrained_ckpt = os.path.join(config.pretrained_ckpt_dir,ViT_configs[config.model_type]["checkpoint"])
  
  # Training.
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 1e-6
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  # config.max_grad_norm = 1.0
  config.label_smoothing = None

  config.num_training_epochs = 30
  
  config.batch_size = 128

  config.steps_per_epoch = (get_aggregated_size(config.dataset_name) // config.batch_size)

  config.rng_seed = 0 

  config.init_head_bias = -10.0 
  config.loss = ml_collections.ConfigDict()
  config.loss.m = 0.0
  config.loss.scale = 16
  config.max_to_keep = 1000

  config.log_eval_steps = config.steps_per_epoch

  # Learning rate.
  base_lr = 1e-3

  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant'
  config.lr_configs.base_learning_rate = base_lr

  config.lr_configs.backbone = ml_collections.ConfigDict()

  #that means for 2 epochs we train only the classifier
  config.lr_configs.backbone.frozen_steps = (
      2 * config.steps_per_epoch
  )

  config.lr_configs.backbone.base_learning_rate = base_lr * 1e-2

  # kNN

  #dir of checkpoints
  config.train_dir = ''
  
  # Logging.

  config.preextracted = False
  #config.preextracted = True

  config.write_summary = True
  
  config.test_pretrained_features = False

  config.extract_only_descrs = False

  config.checkpoint = False 

  config.save_descriptors = True
  #config.save_descriptors = False

  config.debug_eval = False  # Debug mode during eval.
    
  config.eval_dataset_dir = ''
  config.train_dataset_dir = '' 
  
  config.project_feats_knn = True
  #config.project_feats_knn = False

  config.top_k = 5 #top k neighbors to look at
  #config.top_k = 100

  config.knn_start_epoch = 3
  config.knn_end_epoch = 3 #set this to a lower value than start_epoch to not do knn at all

  config.log_csv = False

  config.save_neighbors = False

  config.info_files_dir = ''

  #config.descr_save_path = "."
  config.descr_save_path = None

  return config


