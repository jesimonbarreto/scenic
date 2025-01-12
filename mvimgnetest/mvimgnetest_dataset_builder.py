"""MVImgNet dataset."""

import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
#import tensorflow_datasets.public_api as tfds

import os
import numpy as np
import jax
import jax.numpy as jnp
import re
import random
# Set seeds for reproducibility
seed_value = 42
random.seed(seed_value)  # Fixes seed for Python's random module

# JAX seed, typically used for JAX random operations (if needed)
jax_key = jax.random.PRNGKey(seed_value)


mvimgnet_classes = [
    "bag", "bottle", "washer", "vessel", "train", "telephone", "table", "stove", "sofa", "skateboard", 
    "rifle", "pistol", "remote control", "printer", "flowerpot", "pillow", "piano", "mug", "motorcycle", 
    "microwave", "microphone", "mailbox", "loudspeaker", "laptop", "lamp", "knife", "pot", "helmet", 
    "guitar", "bookshelf", "faucet", "earphone", "display", "dishwasher", "computer keyboard", "clock", 
    "chair", "car", "cap", "can", "camera", "cabinet", "bus", "bowl", "bicycle", "bench", "bed", "bathtub", 
    "basket", "ashcan", "airplane", "umbrella", "plush toy", "toy figure", "towel", "toothbrush", "toy bear", 
    "toy cat", "toy bird", "toy insect", "toy cow", "toy dog", "toy monkey", "toy elephant", "toy fish", 
    "toy horse", "toy sheep", "toy mouse", "toy tiger", "toy rabbit", "toy dragon", "toy snake", "toy chook", 
    "toy pig", "rice cooker", "pressure cooker", "toaster", "dryer", "battery", "curtain", "blackboard eraser", 
    "bucket", "calculator", "candle", "cassette", "cup sleeve", "computer mouse", "easel", "fan", "cookie", 
    "fries", "donut", "coat rack", "guitar stand", "can opener", "flashlight", "hammer", "scissors", "screw driver", 
    "spanner", "hanger", "jug", "fork", "chopsticks", "spoon", "ladder", "ceiling lamp", "wall lamp", "lamp post", 
    "light switch", "mirror", "paper box", "wheelchair", "walking stick", "picture frame", "shower", "toilet", 
    "sink", "power socket", "bagged snacks", "tripod", "selfie stick", "hair dryer", "lipstick", "glasses", 
    "sanitary napkin", "toilet paper", "rockery", "Chinese hot dishes", "root carving", "flower", "book", 
    "pipe PVC metal pipe", "projector", "cabinet air conditioner", "desk air conditioner", "refrigerator", 
    "percussion", "strings", "wind instruments", "balloons", "scarf", "shoe", "skirt", "pants", "clothing", 
    "box", "soccer", "roast duck", "pizza", "ginger", "cauliflower", "broccoli", "cabbage", "eggplant", 
    "pumpkin", "winter melon", "tomato", "corn", "sunflower", "potato", "sweet potato", "Chinese cabbage", 
    "onion", "momordica charantia", "chili", "cucumber", "grapefruit", "jackfruit", "star fruit", "avocado", 
    "shakyamuni", "coconut", "pineapple", "kiwi", "pomegranate", "pawpaw", "watermelon", "apple", "banana", 
    "pear", "cantaloupe", "durian", "persimmon", "grape", "peach", "power strip", "racket", "toy butterfly", 
    "toy duck", "toy turtle", "bath sponge", "glove", "badminton", "lantern", "chestnut", "accessory", "shovel", 
    "cigarette", "stapler", "lighter", "bread", "key", "toothpaste", "swim ring", "watch", "telescope", "eggs", 
    "bun", "guava", "okra", "tangerine", "lotus root", "taro", "lemon", "garlic", "mango", "sausage", "besom", 
    "lock", "ashtray", "conch", "seafood", "hairbrush", "ice cream", "razor", "adhesive hook", "hand warmer", 
    "thermometer", "bell", "sugarcane", "adapter(water pipe)", "calendar", "insecticide", "electric saw", 
    "inflator", "ironmongery", "bulb"
]


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for mvimgnet dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  n_total_pairs = 0


  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(MVImgNet): Specifies the tfds.core.DatasetInfo object
    video_shape = (
        None,
        224,
        224,
        3,
    )
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            #'video': tfds.features.Video(
            #  video_shape,
            #  encoding_format= 'jpeg'),
            'image': tfds.features.Image(encoding_format='jpeg'),
            'label': tfds.features.ClassLabel(names=list(mvimgnet_classes)),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    path = '/mnt/disks/dataset/mvimgnet/data/'
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')  

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "datapath": train_path,
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                "datapath": test_path,
            },
        ),
    ]

  def process_image(self, image_path):
      # Leia o arquivo da imagem
      image = tf.io.read_file(image_path)
      # Decodifique a imagem para um tensor
      image = tf.image.decode_jpeg(image, channels=3)
      # Redimensione a imagem
      image = tf.image.resize(image, [224, 224])
      # Normalize a imagem
      #image = tf.cast(image, tf.float32) / 255.0
      # Converta o tensor para um numpy array
      return image.numpy()

  # Função para extrair o número da sequência do nome do arquivo
  def get_sequence_number(self, path):
      # Usa regex para encontrar o número no nome do arquivo
      match = re.search(r'(\d+)', path)
      if match:
          return int(match.group(1))
      return None
  
  # Função para selecionar n valores aleatórios
  def select_random_values(self, sorted_paths, n):
    # Garantir que n não seja maior que o número de elementos disponíveis
    n = min(n, len(sorted_paths))
    
    # Selecionar n valores aleatórios sem substituição
    random_indices = random.sample(range(len(sorted_paths)), n)
    
    # Retornar os valores correspondentes aos índices selecionados
    random_values = [sorted_paths[i] for i in random_indices]
    
    return random_values


  def _generate_examples(self, datapath):
    """Yields examples."""
    
    datapath, file_path = os.path.split(datapath)
    if not datapath.endswith('/'):
        datapath += '/'
    
    if file_path == 'train':
        file_path = '/mnt/disks/dataset/mvimgnet/train.npz'
    else:
        file_path = '/mnt/disks/dataset/mvimgnet/test.npz'
    
    n = 3


    train_ref = np.load(file_path, allow_pickle=True)
    keys_ref = train_ref.keys()
    

    for label in tf.io.gfile.listdir(datapath):
      if label not in keys_ref:
         print('label')
         print(label)
         print('keys label')
         print(keys_ref)
         continue
      train_class_ref = train_ref[label]
      for obj_var in tf.io.gfile.listdir(os.path.join(datapath, label)):
        if obj_var not in train_class_ref:
           continue
        dir_search = os.path.join(datapath, label, obj_var, 'images', "*.jpg")
        frames_video = tf.io.gfile.glob(dir_search)
        #base_names = [os.path.basename(fpath) for fpath in frames_video]
        id = label+'_'+obj_var

        # Ordena a lista de paths usando o número da sequência como chave
        #frames_video = sorted(frames_video, key=self.get_sequence_number)

        # Seleciona os pares
        samples = self.select_random_values(frames_video, n)
        
        if len(samples) == 0:
           continue
        
        for k, image_path in enumerate(samples):
          img = self.process_image(image_path)
          img = img.astype(jnp.uint8)
          record = {
            "image": img,
            "label": int(label)
          }
          yield str(k)+'_'+id, record

        #ROdar novamente [sem filtro de classes]
        #COnfigurar dataset mvimgnet
        #alterar validação para usar esse
        #Carregar dados
        #executar experimentop