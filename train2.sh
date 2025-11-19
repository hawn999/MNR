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
now=$(date +"%Y-%m-%d-%H-%M-%S")

# for v3 experiments
# 33 r
python -u main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/nvme_data/datasets/resized_datasets_raven/ --gpu 0,1 --fp16 \
            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
            -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
            --classifier-hidreduce 4 --ckpt ckpts/${now}_v3_\
            --reduce-planes 96 --num-hylas 3 --workers 2 --in-channels 1\
            2>&1 | tee log/${now}_v3_33_r_muon_004.txt

# 54 r
python -u main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/nvme_data/datasets/resized_datasets_raven/ --gpu 0,1 --fp16 \
            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
            -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
            --classifier-hidreduce 4 --ckpt ckpts/${now}_v3_\
            --reduce-planes 160 --num-hylas 4 --workers 2 --in-channels 1\
            2>&1 | tee log/${now}_v3_54_r_muon_004.txt
## error
#python -u main_error.py --dataset-name RAVEN --dataset-dir /home/scxhc1/nvme_data/resized_datasets_raven --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_original_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/mlp_raven_error_\
#            --reduce-planes 128 --num-hylas 3 --workers 2 --in-channels 1\
#            2>&1 | tee log/mlp_raven_error.txt


#python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/43_mnr_${now}_\
#            --reduce-planes 128 --num-hylas 3 --workers 2 --in-channels 1\
#            2>&1 | tee log/${now}_43_mnr.txt
#
#python -u main.py --dataset-name RAVEN-FAIR --dataset-dir /home/scxhc1/nvme_data/resized_datasets_raven --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/43_rf_${now}_\
#            --reduce-planes 128 --num-hylas 3 --workers 2 --in-channels 1\
#            2>&1 | tee log/${now}_43_rf.txt

## test MNR
#python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 42 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_mnr_test --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/${now}_43_mnr_v3_seed42_ \
#            --reduce-planes 128 --num-hylas 3 --workers 2 \
#            2>&1 | tee log/${now}_43_mnr_v3_seed42.txt




## 43 MNR eval
#nohup python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0,1 --fp16 \
#               --image-size 80 --seed 3407 --batch-size 128\
#               -a predrnet_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#               --classifier-hidreduce 4 --ckpt ckpts/4GIPRB_3HYLA_eval_ \
#               --reduce-planes 128 --num-hylas 3 --workers 2 \
#               -e --resume /home/scxhc1/MNR_IJCAI25/ckpts/4GIPRB_3HYLA_MNR-predrnet_mnr-prb3-b0.1c0.1-imsz80-wd1e-05-ep200-seed3407/model_best.pth.tar\
#               2>&1 | tee log/4GIPRB_3HYLA_MNR_eval.txt

## 13
#nohup python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0,1 --fp16 \
#               --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#               -a predrnet_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#               --classifier-hidreduce 4 --ckpt ckpts/1GIPRB_3HYLA_ \
#               --reduce-planes 32 --num-hylas 3 --workers 2\
#               2>&1 | tee log/1GIPRB_3HYLA_MNR.txt
## 23
#nohup python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0,1 --fp16 \
#               --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#               -a predrnet_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#               --classifier-hidreduce 4 --ckpt ckpts/2GIPRB_3HYLA_ \
#               --reduce-planes 64 --num-hylas 3 --workers 2\
#               2>&1 | tee log/2GIPRB_3HYLA_MNR.txt
#
## 33
#nohup python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0,1 --fp16 \
#               --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#               -a predrnet_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#               --classifier-hidreduce 4 --ckpt ckpts/3GIPRB_3HYLA_ \
#               --reduce-planes 96 --num-hylas 3 --workers 2\
#               2>&1 | tee log/3GIPRB_3HYLA_MNR.txt

## 42 raven
#nohup python -u main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/nvme_data/resized_datasets_raven --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/4GIPRB_2HYLA_ \
#            --reduce-planes 128 --num-hylas 2 --workers 2\
#            2>&1 | tee log/4GIPRB_2HYLA_RAVEN.txt

## 33RAVEN
#python -u main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/nvme_data/resized_datasets_raven --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/3GIPRB_3HYLA_RAVEN_ \
#            --reduce-planes 96 --num-hylas 3 --workers 2\
#            2>&1 | tee log/3GIPRB_3HYLA_RAVEN.txt


