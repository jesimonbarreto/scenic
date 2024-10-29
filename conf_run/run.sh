cd /home/jesimonbarreto/scenic
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
