#path="/home/Chengtai_Li/文档/final_MNR/table3/SRAVEN_8_8_7w_OOD"
#433
#path="/home/scxhc1/nvme_data/resized_datasets_raven"
#/home/scxhc1/nvme_data/SRAVEN_8_8_7w_OOD
#you
path="/home/scxhc1/nvme_data/MNR_data/datasets/resized_datasets_raven"
#x
#/home/scxdx2/nvme_data/MNR_data/datasets/resized_datasets_raven
#xz
#path="/home/scxhh2/resized_datasets_raven"
#h
#/home/scxhh2/nvme_data/datasets
#/home/scxhh2/nvme_data/cot_raven
# L20

# 43
now=$(date +"%Y-%m-%d-%H-%M-%S")
python -u main.py --dataset-name MNR --dataset-dir /home/scxhc1/nvme_data/datasets --gpu 0,1 --fp16 \
            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
            -a predrnet_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
            --classifier-hidreduce 4 --ckpt ckpts/${now}_v3_ \
            --reduce-planes 128 --num-hylas 3 --workers 2 --in-channels 1 \
            2>&1 | tee log/${now}_predNaN_43_mnr_muon003_clip_norelu_noRoll.txt

#            2>&1 | tee log/${now}_v3_43_mnr_adam_clip5.txt

#now=$(date +"%Y-%m-%d-%H-%M-%S")
#python -u mainNaN.py --dataset-name MNR --dataset-dir /home/scxhc1/nvme_data/datasets --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a prednan_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/${now}_v3_ \
#            --workers 2 --in-channels 1 \
#            2>&1 | tee log/${now}_predNaN_43_mnr.txt
#python main.py --dataset-name MNR --dataset-dir /home/scxhc1/nvme_data/datasets/ --gpu 0,1 \
#               --image-size 80 -a predrnet_mnr \
#               -e --resume /home/scxhc1/MNR_IJCAI25/ckpts/4GIPRB_3HYLA_MNR-predrnet_mnr-prb3-b0.1c0.1-imsz80-wd1e-05-ep200-seed3407/model_best.pth.tar \
#               --show-detail






















