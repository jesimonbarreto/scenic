### test if you need sudo rm -rf /mnt/disks/dataset/dataset/imagenet/mvimgnet/
cd /home/jesimonbarreto/scenic
# sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/mvimgnet/
#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/mvimgnetest/

#sudo rm -rf /mnt/disks/stg_dataset/test
#sudo rm -rf /mnt/disks/stg_dataset/test_video
#sudo rm -rf /mnt/disks/stg_dataset/test_frame
#sudo rm -rf /mnt/disks/stg_dataset/test_video_scratch
#sudo rm -rf /mnt/disks/stg_dataset/test_frame_scratch

sudo -E python -m knn_main --config=configs/my_config_knn_google_dinos.py --workdir=../test_

#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim.py --workdir=/mnt/disks/stg_dataset/test_video
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old_video.py --workdir=../test_

#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_frame.py --workdir=/mnt/disks/stg_dataset/test_frame
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old_frame.py --workdir=../test_

#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_scratch.py --workdir=/mnt/disks/stg_dataset/test_video_scratch
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old_video_scratch.py --workdir=../test_

#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_frame_scratch.py --workdir=/mnt/disks/stg_dataset/test_frame_scratch
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old_frame_scratch.py --workdir=../test_



#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_frame_scratch.py --workdir=/mnt/disks/stg_dataset/test_test_
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old_frame_scratch.py --workdir=../test_

#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_head.py --workdir=/mnt/disks/stg_dataset/test_test
#sudo -E python -m main_dino --config=configs/exp_var/dino_mvimnet_lim_contlearn.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/my_config_knn_google_old_contiLearn.py --workdir=../test_
