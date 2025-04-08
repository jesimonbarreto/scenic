import numpy as np

# Caminho para os arquivos gerados
train_path = '/mnt/disks/stg_dataset/dataset/CO3D/train.npz'
test_path = '/mnt/disks/stg_dataset/dataset/CO3D/test.npz'

# Carrega os arquivos npz
train_data = np.load(train_path, allow_pickle=True)
test_data = np.load(test_path, allow_pickle=True)

print("=== TREINO ===")
for class_name in train_data.files:
    folders = train_data[class_name].tolist()
    print(f"Classe: {class_name} - {len(folders)} exemplos")

print("\n=== TESTE ===")
for class_name in test_data.files:
    folders = test_data[class_name].tolist()
    print(f"Classe: {class_name} - {len(folders)} exemplos")
