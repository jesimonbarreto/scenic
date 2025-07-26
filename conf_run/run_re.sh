### test if you need sudo rm -rf /mnt/disks/dataset/dataset/imagenet/mvimgnet/
cd /home/jesimonbarreto/scenic
#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/mvimgnet/
#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/mvimgnetest/

############MVIMAGNET
#DINO
sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dino_head.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dino_our.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/mvimgnet/knn_eval_dino.py --workdir=../test_ 


#DINO v2
sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dinov2_head.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dinov2_our.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/mvimgnet/knn_eval_dinov2.py --workdir=../test_ 


#TIPS
sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/mvimgnet/tips_head.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/tips_our.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/mvimgnet/knn_eval_tips.py --workdir=../test_ 


############CO3D

#DINO
sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/co3d/dino_head.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/co3d/dino_our.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_dino.py --workdir=../test_ 


#DINO v2
sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/co3d/dinov2_head.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/co3d/dinov2_our.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_dinov2.py --workdir=../test_ 

#TIPS
sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/co3d/tips_head.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/co3d/tips_our.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_tips.py --workdir=../test_ 


