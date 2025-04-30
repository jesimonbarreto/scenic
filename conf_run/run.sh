cd /home/jesimonbarreto/scenic




sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dino_head.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dino_our.py --workdir=/mnt/disks/stg_dataset/test_test2

sudo cp -r /mnt/disks/stg_dataset/test_test2 /mnt/disks/stg_dataset/mvimgnet_video

sudo rm -rf /mnt/disks/stg_dataset/head_2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dino_head_frame.py --workdir=/mnt/disks/stg_dataset/head_2

sudo rm -rf /mnt/disks/stg_dataset/test_test2
sudo -E python -m main_dino --config=configs/final/mvimgnet/dino_frame.py --workdir=/mnt/disks/stg_dataset/test_test2

sudo cp -r /mnt/disks/stg_dataset/test_test2 /mnt/disks/stg_dataset/mvimgnet_frame
