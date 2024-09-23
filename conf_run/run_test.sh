cd /home/jesimonbarreto/scenic
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/dino_imnet1k_base16_last.py --workdir=../test
sudo rm -rf ../test