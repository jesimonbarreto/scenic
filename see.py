import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import CO3Dtest_dataset

classes_co3d = [
    "apple", "backpack", "ball", "banana", "baseballbat", "baseballglove", "bench",
    "bicycle", "book", "bottle", "bowl", "broccoli", "cake", "car", "carrot",
    "cellphone", "chair", "couch", "cup", "donut", "frisbee", "hairdryer", "handbag",
    "hotdog", "hydrant", "keyboard", "kite", "laptop", "microwave", "motorcycle",
    "mouse", "orange", "parkingmeter", "pizza", "plant", "remote", "sandwich",
    "skateboard", "stopsign", "suitcase", "teddybear", "toaster", "toilet", "toybus",
    "toyplane", "toytrain", "toytruck", "tv", "umbrella", "vase"
]



# Carrega o dicionário com os matches
data = np.load('/home/jesimonbarreto/Documents/mestrado/plot_exp/match_results.npz', allow_pickle=True)
results_dict = {k: data[k].item() for k in data}

# Carrega os splits
print("🔄 Carregando validation...")
val_ds = tfds.load('co3dtest', split='validation', data_dir='/home/jesimonbarreto/Documents/mestrado/plot_exp', shuffle_files=False, as_supervised=False)
print("🔄 Carregando train...")
train_ds = tfds.load('co3dtest', split='train', data_dir='/home/jesimonbarreto/Documents/mestrado/plot_exp', shuffle_files=False, as_supervised=False)

val_ds = tfds.as_numpy(val_ds)
train_ds = tfds.as_numpy(train_ds)

# Indexando por índice string + armazenando também a classe
def build_indexed_dict_with_class(dataset):
    indexed = {}
    for ex in dataset:
        index = ex['index'].decode() if isinstance(ex['index'], bytes) else ex['index']
        class_name = ex['label'].decode() if isinstance(ex['label'], bytes) else ex['label']
        indexed[index] = {'image': ex['image'], 'label': class_name}
    return indexed

val_dict = build_indexed_dict_with_class(val_ds)
train_dict = build_indexed_dict_with_class(train_ds)

# Extrai o índice da string no formato TFRecord__index
def extract_index(full_id):
    return full_id.split('__')[-1]

# Plotagem com classe no título
def plot_case(val_id, correct_id, incorrect_id):
    val_index = extract_index(val_id)
    correct_index = extract_index(correct_id)
    incorrect_index = extract_index(incorrect_id)

    val_ex = val_dict.get(val_index)
    correct_ex = train_dict.get(correct_index)
    incorrect_ex = train_dict.get(incorrect_index)

    if val_ex is None or correct_ex is None or incorrect_ex is None:
        print(f"⚠️ Um ou mais índices não encontrados: {val_index}, {correct_index}, {incorrect_index}")
        return

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(val_ex['image'])
    axs[0].set_title(f"🔍 Validation\nClass: {classes_co3d[int(val_ex['label'])]}")
    axs[0].axis('off')

    axs[1].imshow(correct_ex['image'])
    axs[1].set_title(f"✅ Correct Match\nClass: {classes_co3d[int(correct_ex['label'])]}")
    axs[1].axis('off')

    axs[2].imshow(incorrect_ex['image'])
    axs[2].set_title(f"❌ Incorrect Match\nClass: {classes_co3d[int(incorrect_ex['label'])]}")
    axs[2].axis('off')

    plt.suptitle(f"Index: {val_index}", fontsize=14)
    plt.tight_layout()
    plt.show()

# Plotar todos os casos
for val_id, match in results_dict.items():
    print(val_id)
    print(match)
    plot_case(val_id, match['correct'], match['incorrect'])
