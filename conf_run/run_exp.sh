### test if you need sudo rm -rf /mnt/disks/dataset/dataset/imagenet/mvimgnet/
cd /home/jesimonbarreto/scenic
sudo rm -rf /mnt/disks/dataset/dataset/imagenet/mvimgnet/
sudo rm -rf /mnt/disks/dataset/test
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim.py --workdir=/mnt/disks/dataset/test
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_
# sudo rm -rf /mnt/disks/dataset/test
# sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim.py --workdir=/mnt/disks/dataset/test
# sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_
# sudo rm -rf /mnt/disks/dataset/test
# sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim.py --workdir=/mnt/disks/dataset/test
# sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_
sudo rm -rf ../test_
sudo -E python -m knn_main --config=configs/my_config_knn_google_dinos.py --workdir=../test_
## sudo rm -rf ../test_
## sudo -E python -m knn_main --config=configs/my_config_knn_google_dinob.py --workdir=../test_

sudo rm -rf /mnt/disks/dataset/test
sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim.py --workdir=/mnt/disks/dataset/test
sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_