import os
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import imageio
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

data = np.load('/home/jesimonbarreto/Documents/mestrado/plot_exp/match_results_triple.npz', allow_pickle=True)
results_dict = {k: data[k].item() for k in data}

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

# Novo: função para salvar imagens em pastas nomeadas
def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path, image)

# Substituto de `plot_case` para salvar imagens
def save_case_images(val_id, correct_id, incorrect_base_id, incorrect_img_id):
    val_index = extract_index(val_id)
    correct_index = extract_index(correct_id)
    incorrect_base_index = extract_index(incorrect_base_id)
    incorrect_img_index = extract_index(incorrect_img_id)

    val_ex = val_dict.get(val_index)
    correct_ex = train_dict.get(correct_index)
    incorrect_base_ex = train_dict.get(incorrect_base_index)
    incorrect_img_ex = train_dict.get(incorrect_img_index)

    if not all([val_ex, correct_ex, incorrect_base_ex, incorrect_img_ex]):
        print(f"⚠️ Índices não encontrados: {val_index}, {correct_index}, {incorrect_base_index}, {incorrect_img_index}")
        return

    base_path = os.path.join('/home/jesimonbarreto/Documents/mestrado/plot_exp/samples', val_index)
    
    save_image(val_ex['image'], os.path.join(base_path, f'validation_class_{classes_co3d[int(val_ex["label"])]}.png'))
    save_image(correct_ex['image'], os.path.join(base_path, f'correct_class_{classes_co3d[int(correct_ex["label"])]}.png'))
    save_image(incorrect_img_ex['image'], os.path.join(base_path, f'incorrect_class_image_{classes_co3d[int(incorrect_img_ex["label"])]}.png'))
    save_image(incorrect_base_ex['image'], os.path.join(base_path, f'incorrect_class_base_{classes_co3d[int(incorrect_base_ex["label"])]}.png'))

    # Comentado: visualização com matplotlib
    # fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    # axs[0].imshow(val_ex['image'])
    # axs[0].set_title(f"🔍 Validation\nClass: {classes_co3d[int(val_ex['label'])]}")
    # axs[0].axis('off')
    # axs[1].imshow(correct_ex['image'])
    # axs[1].set_title(f"✅ Correct\nClass: {classes_co3d[int(correct_ex['label'])]}")
    # axs[1].axis('off')
    # axs[2].imshow(incorrect_base_ex['image'])
    # axs[2].set_title(f"❌ Incorrect Base\nClass: {classes_co3d[int(incorrect_base_ex['label'])]}")
    # axs[2].axis('off')
    # axs[3].imshow(incorrect_img_ex['image'])
    # axs[3].set_title(f"❌ Incorrect Img\nClass: {classes_co3d[int(incorrect_img_ex['label'])]}")
    # axs[3].axis('off')
    # plt.suptitle(f"Index: {val_index}", fontsize=14)
    # plt.tight_layout()
    # plt.show()

# Salvar imagens para todos os casos
for val_id, match in results_dict.items():
    print(f"💾 Salvando amostra: {val_id}")
    save_case_images(val_id, match['correct'], match['incorrect_base'], match['incorrect_img'])
