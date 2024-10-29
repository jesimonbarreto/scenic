cd /home/jesimonbarreto/scenic
sudo -E python -m main_dino --config=configs/dino_imnet1k_base16_last.py --workdir=../random_limiclass
sudo -E python -m knn_main --config=configs/my_config_knn.py --workdir=../test
