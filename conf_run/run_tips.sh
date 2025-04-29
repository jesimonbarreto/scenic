### test if you need sudo rm -rf /mnt/disks/dataset/dataset/imagenet/mvimgnet/
cd /home/jesimonbarreto/scenic
#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/mvimgnet/
#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/mvimgnetest/


#tips pretrained
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_tips_base.py --workdir=../test_

#frame
sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/co3d/tips_head_frame.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/co3d/tips_frame.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_tips.py --workdir=../test_

#video
sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/co3d/tips_head.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/co3d/tips_our.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_tips.py --workdir=../test_


#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/last_exp/dino_mvimnet_lim_head_all_b.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/last_exp/dino_mvimnet_lim_contlearn_two_h2_all_proj_b.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/last_exp/my_config_knn_google_old_contiLearn_1_b.py --workdir=../test_

#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/last_exp/dino_mvimnet_lim_head_all_b_frame.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/last_exp/dino_mvimnet_lim_contlearn_two_h2_all_proj_b_frame.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/last_exp/my_config_knn_google_old_contiLearn_1_b.py --workdir=../test_

##########

#sudo -E python -m knn_main --config=configs/last_exp/my_config_knn_google_co3d_dinob.py --workdir=../test_

#sudo rm -rf /mnt/disks/stg_dataset/dataset/imagenet/co3d/
#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/last_exp/dino_co3d_lim_head_all_b.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/last_exp/dino_co3d_lim_contlearn_two_h2_all_proj_b.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/last_exp/my_config_knn_google_old_contiLearn_1_b_co3d.py --workdir=../test_
#sudo rm -rf /mnt/disks/stg_dataset/head_2
#sudo -E python -m main_dino --config=configs/last_exp/dino_co3d_lim_head_all_b_frame.py --workdir=/mnt/disks/stg_dataset/head_2

#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/last_exp/dino_mvimnet_lim_contlearn_two_h2_all_proj_b_frame.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/last_exp/my_config_knn_google_old_contiLearn_1_b_co3d.py --workdir=../test_


#without training head - Dino v1 - CO3D
sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/co3d/dino_nohead.py --workdir=/mnt/disks/stg_dataset/test_test2
sudo -E python -m knn_main --config=configs/final/co3d/knn_eval_dino.py --workdir=../test_

#without training head - Dino v1 - MVIMGNET
#sudo rm -rf /mnt/disks/stg_dataset/test_test2
#sudo -E python -m main_dino --config=configs/final/mvimgnet/dino_nohead.py --workdir=/mnt/disks/stg_dataset/test_test2
#sudo -E python -m knn_main --config=configs/final/mvimgnet/knn_eval_dino.py --workdir=../test_
