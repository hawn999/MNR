# nohup python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0 --fp16 \
#                --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                --resume ckpts/save/pretrain-MNR-predrnet_mnr-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed3407/checkpoint.pth.tar \
#                > output_mnr.txt &

#nohup python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0 --fp16 \
#               --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#               -a predrnet_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#               --classifier-hidreduce 4 --ckpt ckpts/ \
#               > output_mnr.txt &


#nohup python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0,1 --fp16 \
#               --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#               -a predrnet_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#               --classifier-hidreduce 4 --ckpt ckpts/test2_ \
#               --reduce-planes 128 --num-hylas 3 \
#               2>&1 | tee log/output_mnr_test2.txt

#python -u main.py --dataset-name MNR --dataset-dir ./ --gpu 0,1 --fp16 \
#               --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#               -a predrnet_mnr --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#               --classifier-hidreduce 4 --ckpt ckpts/5GIPRB_3HYLA_ \
#               --reduce-planes 160 --num-hylas 3 \
#               --resume /home/scxhc1/RAVEN-pred-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed3407/checkpoint.pth.tar \
#               2>&1 | tee log/5GIPRB_3HYLA_MNR.txt


#python -u main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/MNR_IJCAI25/dataset/datasets --gpu 0,1 --fp16 \
#               --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#               -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#               --classifier-hidreduce 4 --ckpt ckpts/1GIPRB_4HYLA_ \
#               --reduce-planes 32 --num-hylas 4 --workers 2\
#               --resume /home/scxhc1/RAVEN-pred-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed3407/checkpoint.pth.tar \
#               2>&1 | tee log/1GIPRB_4HYLA_RAVEN.txt

# 5 4 raven
python -u main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/MNR_IJCAI25/dataset/datasets --gpu 0,1 --fp16 \
               --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
               -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
               --classifier-hidreduce 4 --ckpt ckpts/5GIPRB_4HYLA_check_resume_ \
               --reduce-planes 160 --num-hylas 4 --workers 2\
               --resume /home/scxhc1/RAVEN-pred-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed3407/checkpoint.pth.tar \
               2>&1 | tee log/5GIPRB_4HYLA_RAVEN_check_resume.txt

## 5 2
#python -u main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/MNR_IJCAI25/dataset/datasets --gpu 0,1 --fp16 \
#               --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#               -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#               --classifier-hidreduce 4 --ckpt ckpts/2GIPRB_4HYLA_ \
#               --reduce-planes 64 --num-hylas 4 --workers 2\
#               --resume /home/scxhc1/RAVEN-pred-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed3407/checkpoint.pth.tar \
#               2>&1 | tee log/2GIPRB_4HYLA_RAVEN.txt



