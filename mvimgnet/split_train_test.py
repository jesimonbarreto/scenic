import os
import random
import numpy as np

# Caminho do dataset
dataset_path = '/mnt/disks/dataset/mvimgnet/data/'

# Configurar a seed para garantir a reprodução
SEED = 42
random.seed(SEED)

# Função para listar todos os vídeos por classe
def get_video_list():
    video_dict = {}
    for classe in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, classe)
        if os.path.isdir(class_path):
            video_dict[classe] = []
            for video in os.listdir(class_path):
                video_path = os.path.join(class_path, video, 'images')
                if os.path.isdir(video_path):  # Verifica se o diretório 'images' existe
                    video_dict[classe].append(video)
    return video_dict

# Função para dividir vídeos de forma balanceada em treino e teste
def split_train_test_balanceado(video_dict, train_ratio=0.75):
    train_videos = {}
    test_videos = {}

    for classe, videos in video_dict.items():
        # Aleatoriamente divide os vídeos dentro da classe
        random.shuffle(videos)  # Usa a seed configurada
        split_index = int(len(videos) * train_ratio)

        # Divide os vídeos em treino e teste, balanceadamente
        train_videos[classe] = videos[:split_index]
        test_videos[classe] = videos[split_index:]

    return train_videos, test_videos

# Função para salvar as referências dos vídeos no formato .npz
def save_video_references_npz(train_videos, test_videos, train_file='train.npz', test_file='test.npz'):
    # Convertendo para formato numpy.array (listas de vídeos)
    np.savez(train_file, **train_videos)
    np.savez(test_file, **test_videos)

# Principal
def main():
    # Passo 1: Obter lista de vídeos por classe
    video_dict = get_video_list()

    # Passo 2: Dividir os vídeos de forma balanceada em treino e teste
    train_videos, test_videos = split_train_test_balanceado(video_dict)

    # Passo 3: Salvar referências em formato .npz
    save_video_references_npz(train_videos, test_videos)

    print("Divisão balanceada de vídeos concluída e salva no formato .npz!")

if __name__ == '__main__':
    main()
