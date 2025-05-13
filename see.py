import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from PIL import Image
import CO3Dtest_dataset

# Lista de classes
classes_co3d = [
    "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove", "bench",
    "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
    "cellphone", "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
    "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave", "motorcycle",
    "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
    "skateboard", "stopsign", "suitcase", "teddybear", "toaster", "toilet", "toybus",
    "toyplane", "toytrain", "toytruck", "tv", "umbrella", "vase"
]

# Diretório base de saída
base_out_dir = '/home/jesimonbarreto/Documents/mestrado/plot_exp/samples'
os.makedirs(base_out_dir, exist_ok=True)

# Carrega o dicionário com os matches
data = np.load('/home/jesimonbarreto/Documents/mestrado/plot_exp/match_results_correct_base_50.npz', allow_pickle=True)
results_dict = {k: data[k].item() for k in data}

# Carrega os splits
print("🔄 Carregando validation...")
val_ds = tfds.load('co3dtest', split='validation', data_dir='/home/jesimonbarreto/Documents/mestrado/plot_exp', shuffle_files=False, as_supervised=False)
print("🔄 Carregando train...")
train_ds = tfds.load('co3dtest', split='train', data_dir='/home/jesimonbarreto/Documents/mestrado/plot_exp', shuffle_files=False, as_supervised=False)

val_ds = tfds.as_numpy(val_ds)
train_ds = tfds.as_numpy(train_ds)

def build_indexed_dict_with_class(dataset):
    indexed = {}
    for ex in dataset:
        index = ex['index'].decode() if isinstance(ex['index'], bytes) else ex['index']
        class_name = ex['label'].decode() if isinstance(ex['label'], bytes) else ex['label']
        indexed[index] = {'image': ex['image'], 'label': class_name}
    return indexed

val_dict = build_indexed_dict_with_class(val_ds)
train_dict = build_indexed_dict_with_class(train_ds)

def extract_index(full_id):
    return full_id.split('__')[-1]

# Função para salvar imagem
def save_image(img_array, path):
    img = Image.fromarray(img_array)
    img.save(path)

# Para cada amostra, salva as imagens — mas apenas as 50 primeiras
for i, (val_id, match) in enumerate(results_dict.items()):
    if i >= 50:
        break

    val_index = extract_index(val_id)
    correct_index = extract_index(match['correct'])
    incorrect_index = extract_index(match['incorrect_base'])

    val_ex = val_dict.get(val_index)
    correct_ex = train_dict.get(correct_index)
    incorrect_ex = train_dict.get(incorrect_index)

    if val_ex is None or correct_ex is None or incorrect_ex is None:
        print(f"⚠️ Um ou mais índices não encontrados: {val_index}, {correct_index}, {incorrect_index}")
        continue

    val_class = classes_co3d[int(val_ex['label'])]
    correct_class = classes_co3d[int(correct_ex['label'])]
    incorrect_class = classes_co3d[int(incorrect_ex['label'])]

    sample_dir = os.path.join(base_out_dir, val_index)
    os.makedirs(sample_dir, exist_ok=True)

    save_image(val_ex['image'], os.path.join(sample_dir, f"validation_{val_class}.png"))
    save_image(correct_ex['image'], os.path.join(sample_dir, f"correct_{correct_class}.png"))
    save_image(incorrect_ex['image'], os.path.join(sample_dir, f"incorrect_{incorrect_class}.png"))

    # Comentado: plotagem das imagens
    # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    # axs[0].imshow(val_ex['image'])
    # axs[0].set_title(f"🔍 Validation\nClass: {val_class}")
    # axs[0].axis('off')
    # axs[1].imshow(correct_ex['image'])
    # axs[1].set_title(f"✅ Correct\nClass: {correct_class}")
    # axs[1].axis('off')
    # axs[2].imshow(incorrect_ex['image'])
    # axs[2].set_title(f"❌ Incorrect\nClass: {incorrect_class}")
    # axs[2].axis('off')
    # plt.suptitle(f"Sample: {val_index}")
    # plt.tight_layout()
    # plt.show()
