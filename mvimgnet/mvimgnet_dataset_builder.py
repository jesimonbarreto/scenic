"""MVImgNet dataset."""

import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils.lazy_imports_utils import tensorflow as tf
#import tensorflow_datasets.public_api as tfds

import os
import numpy as np

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
            'image': tfds.features.Video(
              video_shape,
              encoding_format= 'jpeg'),
            'label': tfds.features.ClassLabel(names=list(mvimgnet_classes)),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('video', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(MVImgNet): Downloads the data and defines the splits
    #path = dl_manager.download_and_extract('https://todo-data-url')

    path = os.path.join('/mnt/disks/persist/dataset/mvimgnet','1.0.0')
    

    # TODO(MVImgNet): Returns the Dict[split names, Iterator[Key, Example]]
  
    '''dirname = self.builder_config.dirname
    url = _URL_PREFIX + "{}.tgz".format(dirname)
    path = dl_manager.download_and_extract(url)
    train_path = os.path.join(path, dirname, "train")
    val_path = os.path.join(path, dirname, "val")'''

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "datapath": path,
            },
        )
    ]

  '''def _generate_examples(self, datapath):
    """Yields examples."""
    for label in tf.io.gfile.listdir(datapath):
      for obj_var in tf.io.gfile.listdir(os.path.join(datapath, label)):
        for fpath in tf.io.gfile.glob(os.path.join(datapath, label, obj_var, "*.jpg")):
          fname = os.path.basename(fpath)
          record = {
              "image": fpath,
              "label": mvimgnet_classes[label],
              "label_number": label,
              "obj_var": obj_var
          }
          yield fname, record
  '''


  def _generate_examples(self, datapath):
    """Yields examples."""
    for label in tf.io.gfile.listdir(datapath):
      print(f" label {label}")
      for obj_var in tf.io.gfile.listdir(os.path.join(datapath, label)):
        print(f" obj_var {obj_var}")
        dir_search = os.path.join(datapath, label, obj_var,'images', "*.jpg")
        frames_video = tf.io.gfile.glob(dir_search)
        #base_names = [os.path.basename(fpath) for fpath in frames_video]
        print(f" base names {frames_video[:2]+frames_video[:2]}")
        id = label+'_'+obj_var
        print(f" id {id}")
        def process_image(image_path):
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
        
        video_ = []
        for image_path in frames_video[:2]:
          img = process_image(image_path)
          img = img.astype(np.uint8)
          video_.append(np.expand_dims(img, axis=0))
        
        video_ = np.concatenate(video_)
        print(f' video {video_.shape} type {video_.dtype} max {np.max(video_)}')

        record = {
          "image": video_,
          "label": int(label)
        }
        yield id, record