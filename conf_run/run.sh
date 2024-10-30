cd /home/jesimonbarreto/scenic
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/dino_mv_transf_no.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/dino_mv_transf_blur.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/dino_mv_transf_gray.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/dino_mv_transf_jit.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/dino_mv_transf_sol.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
