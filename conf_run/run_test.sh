### test if you need sudo rm -rf /mnt/disks/dataset/dataset/imagenet/mvimgnet/
cd /home/jesimonbarreto/scenic
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim.py --workdir=../test
sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim.py --workdir=../test
sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim.py --workdir=../test
sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim.py --workdir=../test
sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim.py --workdir=../test
sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_

sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_frame.py --workdir=../test
sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_frame.py --workdir=../test
sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_frame.py --workdir=../test
sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_frame.py --workdir=../test
sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_frame.py --workdir=../test
sudo -E python -m knn_main --config=configs/my_config_knn_google_old.py --workdir=../test_
sudo rm -rf ../test