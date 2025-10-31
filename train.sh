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
#/home/scxhh2/nvme_data/cot_raven
now=$(date +"%Y-%m-%d-%H-%M-%S")

# prb
python -u main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/i-raven-cot --gpu 0,1 --fp16 \
            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
            -a cot_prb --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
            --classifier-hidreduce 4 --ckpt ckpts/cot_prb_ir_adam_${now}\
            --workers 2 --in-channels 1\
            2>&1 | tee log/${now}_cot_prb_ir_adam_mask34.txt

## v3
#python -u main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/i-raven-cot --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a cot_slot --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/cot_slot_ir_adam_${now}\
#            --reduce-planes 128 --num-hylas 3 --workers 2 --in-channels 1\
#            2>&1 | tee log/${now}_cot_slot_ir_adam.txt

#python -u main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/nvme_data/datasets/resized_datasets_raven --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/43_r_muon_${now}\
#            --reduce-planes 128 --num-hylas 3 --workers 2 --in-channels 1\
#            2>&1 | tee log/${now}_resized_43_r_muon.txt

## hpai
#python -u main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/i-raven-cot --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a cot_hpai --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/cot_hpai_ir_adam_${now}\
#            --workers 2 --in-channels 1\
#            2>&1 | tee log/${now}_cot_hpai_ir_adam.txt
## error
#python -u main_error.py --dataset-name RAVEN --dataset-dir /home/scxhc1/nvme_data/resized_datasets_raven --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_original_raven --num-extra-stages 5 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/mlp_raven_error_16_5_${now}_\
#            --reduce-planes 128 --num-hylas 3 --workers 2 --in-channels 1\
#            2>&1 | tee log/mlp_raven_error_16_5_${now}.txt

#python -u main_error.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/nvme_data/resized_datasets_raven --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_original_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/mlp_ir_error_${now}_\
#            --reduce-planes 128 --num-hylas 3 --workers 2 --in-channels 1\
#            2>&1 | tee log/mlp_ir_error_${now}.txt
#python -u main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/nvme_data/resized_datasets_raven --gpu 1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_raven_graph --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/k3_raven_\
#            --reduce-planes 96 --num-hylas 3 --workers 2 --in-channels 1\
#            2>&1 | tee log/${now}_k3_raven_.txt
## sraven
#python -u main.py --dataset-name SRAVEN --dataset-dir /home/scxhc1/nvme_data/SRAVEN_8_8_7w_OOD --gpu 1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_sraven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/k3_sraven_01_20\
#            --reduce-planes 96 --num-hylas 3 --workers 2 --in-channels 1\
#            2>&1 | tee log/k3_sraven_01_20.txt

#            --resume /home/scxhc1/MNR_IJCAI25/resume/I-RAVEN-pred-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed3407/model_best.pth.tar \
#            2>&1 | tee log/k3_iraven_resume_.txt

## test MNR
#python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_mnr_test --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/k3_mnr_ \
#            --reduce-planes 96 --num-hylas 3 --workers 2 --show-detail \
#            2>&1 | tee log/k3_mnr_01.txt

#python -u main.py --dataset-name SRAVEN --dataset-dir /home/scxhc1/SRAVEN_8_8_7w_OOD --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 616 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a darr_sraven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/darr_sraven_ \
#            --reduce-planes 32 --num-hylas 3 --workers 2\
#            2>&1 | tee log/darr_sraven.txt
##            2>&1 | tee log/${now}_k1_sraven_noDAG.txt

## RVP
#python -u main.py --dataset-name RPV --dataset-dir /home/scxhc1/nvme_data --gpu 0,1 --fp16 \
#                --image-size 80 --epochs 200 --seed 3407 --batch-size 16 --lr 0.001 --wd 1e-5 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
#                --workers 2 --reduce-planes 96\
#                2>&1 | tee log/RVP_.txt


#python -u main.py --dataset-name RAVEN --dataset-dir $path --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 616 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_mnr_test --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/43_raven_v1 \
#            --reduce-planes 128 --num-hylas 3 --workers 2 \
#            2>&1 | tee log/43_raven_v1.txt




#PYTHONUNBUFFERED=1 python main.py --dataset-name SRAVEN --dataset-dir /home/scxhc1/nvme_data/SRAVEN_8_8_7w_OOD --gpu 0,1 --fp16 \
#               --image-size 80 --epochs 100 --seed 12345 --batch-size 256 --lr 0.001 --wd 1e-5 \
#               -a predrnet_sraven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#               --ckpt ckpts/sraven_slot_test \
#               --reduce-planes 128 --num-hylas 3 \
#               2>&1 | tee log/sraven_slot_test.txt

#python -u main.py --dataset-name RAVEN --dataset-dir $path --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/evaluate \
#            --resume /home/scxhc1/MNR_IJCAI25/ckpts/4GIPRB_3HYLA_no_resume_youRAVEN-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep200-seed3407/model_best.pth.tar\
#            --reduce-planes 128 --num-hylas 3 --workers 2 --evaluate
## 43RAVEN
#python -u main.py --dataset-name RAVEN --dataset-dir $path --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/hcv_4GIPRB_3HYLA_RAVEN_time_ \
#            --reduce-planes 128 --num-hylas 3 --workers 2 \
#            2>&1 | tee log/hcv_4GIPRB_3HYLA_RAVEN-FAIR_.txt


## 43RAVEN
#python -u main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/nvme_data/resized_datasets_raven --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 512 --lr 0.001 --wd 1e-5 \
#            -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/4GIPRB_3HYLA_no_resume_you \
#            --reduce-planes 128 --num-hylas 3 --workers 2\
#            --resume /home/scxhc1/992_4GIPRB_3HYLA_RAVEN-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep200-seed616/model_best.pth.tar \
#            --evaluate \
#            2>&1 | tee log/4GIPRB_3HYLA_no_resume.txt

# 53 rf
#python -u main.py --dataset-name RAVEN-FAIR --dataset-dir /home/scxhc1/nvme_data/resized_datasets_raven --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/5GIPRB_3HYLA_RAVEN-FAIR_ \
#            --reduce-planes 160 --num-hylas 3 --workers 2\
#            2>&1 | tee log/5GIPRB_3HYLA_RAVEN-FAIR.txt

## MNRs
#nohup python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0,1 --fp16 \
#               --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#               -a predrnet_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#               --classifier-hidreduce 4 --ckpt ckpts/4GIPRB_3HYLA_ \
#               --reduce-planes 128 --num-hylas 3 --workers 2\
#               2>&1 | tee log/4GIPRB_3HYLA_MNR.txt
## 41
#nohup python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0,1 --fp16 \
#               --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#               -a predrnet_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#               --classifier-hidreduce 4 --ckpt ckpts/4GIPRB_1HYLA_ \
#               --reduce-planes 128 --num-hylas 1 --workers 2\
#               2>&1 | tee log/4GIPRB_1HYLA_MNR.txt
## 42
#nohup python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0,1 --fp16 \
#               --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#               -a predrnet_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#               --classifier-hidreduce 4 --ckpt ckpts/4GIPRB_2HYLA_ \
#               --reduce-planes 128 --num-hylas 2 --workers 2\
#               2>&1 | tee log/4GIPRB_2HYLA_MNR.txt


















