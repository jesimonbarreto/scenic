import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import CO3Dtest_dataset

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

# Função auxiliar para montar o dicionário indexado por ID completo
'''def build_indexed_dict(dataset, split_name):
    indexed = {}
    for i, ex in enumerate(dataset):
        # Monte a chave conforme o padrão "co3dtest-split.tfrecord-XXXX-of-YYYY__ZZZZ"
        index = ex.get('index', i)
        tfrecord = f"{split_name}.tfrecord-00000-of-00001"  # ajuste se necessário
        full_id = f"{tfrecord}__{index}"
        indexed[full_id] = ex['image']
    return indexed
'''
def build_indexed_dict(dataset, split_name, tfrecord_prefix="co3dtest", total_shards=1, shard_format_width=5):
    """
    Constrói um dicionário indexado com a chave no formato:
    'co3dtest-split.tfrecord-XXXXX-of-YYYYY__INDEX'

    Args:
        dataset: dataset carregado via tfds.
        split_name: 'train' ou 'validation'.
        tfrecord_prefix: prefixo, como 'co3dtest'.
        total_shards: número total de arquivos TFRecord no split.
        shard_format_width: número de dígitos no nome dos arquivos (ex: 5 para '00000').

    Returns:
        dicionário mapeando ID para imagem.
    """
    indexed = {}
    examples_per_shard = len(dataset) // total_shards

    for i, ex in enumerate(dataset):
        index = ex.get('index', i)

        shard_id = i // examples_per_shard
        shard_str = str(shard_id).zfill(shard_format_width)
        total_str = str(total_shards).zfill(shard_format_width)

        tfrecord_name = f"{tfrecord_prefix}-{split_name}.tfrecord-{shard_str}-of-{total_str}"
        full_id = f"{tfrecord_name}__{index}"

        indexed[full_id] = ex['image']

    return indexed

# Monte os dicionários indexados
val_dict = build_indexed_dict(val_ds, split_name='validation', total_shards=1) #build_indexed_dict(val_ds, "co3dtest-validation")
train_dict = build_indexed_dict(train_ds, split_name='train', total_shards=2) #build_indexed_dict(train_ds, "co3dtest-train")

# Função para plotar
def plot_case(val_id, correct_id, incorrect_id):
    val_img = val_dict.get(val_id)
    correct_img = train_dict.get(correct_id)
    incorrect_img = train_dict.get(incorrect_id)

    if val_img is None or correct_img is None or incorrect_img is None:
        print(f"⚠️ Um ou mais IDs não encontrados: {val_id}, {correct_id}, {incorrect_id}")
        return

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(val_img)
    axs[0].set_title("🔍 Validation")
    axs[0].axis('off')

    axs[1].imshow(correct_img)
    axs[1].set_title("✅ Correct Match")
    axs[1].axis('off')

    axs[2].imshow(incorrect_img)
    axs[2].set_title("❌ Incorrect Match")
    axs[2].axis('off')

    plt.suptitle(f"Sample: {val_id}", fontsize=14)
    plt.tight_layout()
    plt.show()

# Plotar todos os casos
for val_id, match in results_dict.items():
    print(val_id)
    print(match)
    plot_case(val_id, match['correct'], match['incorrect'])
