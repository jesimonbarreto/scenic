cd /home/jesimonbarreto/scenic
#loss normal - Dino v1 - CO3D
sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/co3d/dino_head.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/co3d/dino_our.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_dino.py --workdir=../test_
