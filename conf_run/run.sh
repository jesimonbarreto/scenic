cd /home/jesimonbarreto/scenic
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/video/dino_video_flip.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/video/dino_video_crop.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/video/dino_video_sol.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/video/dino_video_jit.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/video/dino_video_gray.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/video/dino_video_blur.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/video/dino_video_no.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/frame/dino_video_flip.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/frame/dino_video_crop.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/frame/dino_video_sol.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/frame/dino_video_jit.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/frame/dino_video_gray.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/frame/dino_video_blur.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test
sudo -E python -m main_dino --config=configs/transfor_var/frame/dino_video_no.py --workdir=../test
sudo -E python -m knn_main --config=configs/transfor_var/my_config_knn_google.py --workdir=../test_
sudo rm -rf ../test