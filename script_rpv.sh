#python main.py --dataset-name RPV --dataset-dir /home/scxhc1/MNR_IJCAI25/dataset/datasets --gpu 0,1 --fp16 \
#                --image-size 80 --epochs 100 --seed 3407 --batch-size 16 --lr 0.001 --wd 1e-5 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
#                --classifier-hidreduce 4 --ckpt ckpts/43_rpv \
#                --reduce-planes 128 --num-hylas 3 \
#                2>&1 | tee log/43_rpv_test.log

#python -u main.py --dataset-name RPV --dataset-dir /home/scxhc1/MNR_IJCAI25/dataset/datasets --gpu 0,1 --fp16 \
#                --image-size 80 --epochs 100 --seed 3407 --batch-size 16 --lr 0.001 --wd 1e-5 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
#                --reduce-planes 96 --num-hylas 3 --workers 2 \
#                2>&1 | tee log/k3_RVP.txt


python -u main.py --dataset-name RPV --dataset-dir /home/scxhc1/MNR_IJCAI25/dataset/datasets --gpu 0,1 --fp16 \
                --image-size 80 --epochs 100 --seed 3407 --batch-size 64 --lr 0.001 --wd 1e-5 \
                -a hpai_raven --num-extra-stages 3 --block-drop 0.55 --classifier-drop 0.5 --in-channels 3 \
                --classifier-hidreduce 4 --ckpt ckpts/ \
                2>&1 | tee log/hpai_RVP_055.txt

#python -u main.py --dataset-name RPV --dataset-dir /home/scxhc1/MNR_IJCAI25/dataset/datasets --gpu 0,1 --fp16 \
#                --image-size 80 --epochs 100 --seed 3407 --batch-size 32 --lr 0.001 --wd 1e-5 \
#                -a hpai_pric_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                > ablation-main-hpai_pric_raven-200rpv.log &

#python -u main.py --dataset-name RPV --dataset-dir /home/scxhc1/MNR_IJCAI25/dataset/datasets --gpu 0,1 --fp16 \
#                --image-size 80 --epochs 100 --seed 3407 --batch-size 32 --lr 0.001 --wd 1e-5 \
#                -a darr_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                > ablation-main-darr_raven-rpv.log &
#python -u main.py --dataset-name RPV --dataset-dir /home/scxhc1/MNR_IJCAI25/dataset/datasets --gpu 0 --fp16 \
#                --image-size 80 --epochs 100 --seed 3407 --batch-size 32 --lr 0.001 --wd 1e-5 \
#                -a hcvarr_rpv --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                > ablation-main-hcvarr_rpv.log &

#nohup python -u main.py --dataset-name RPV --dataset-dir /home/scxhc1/MNR_IJCAI25/dataset/datasets --gpu 0 --fp16 \
#               --image-size 80 --epochs 500 --seed 3407 --batch-size 32 --lr 0.001 --wd 1e-5 \
#               -a hcv_pric_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
#               --classifier-hidreduce 4 --ckpt ckpts/ \
#               > ablation-main-hcv_pric-rpv500.log &

#python -u main.py --dataset-name RPV --dataset-dir /home/scxhc1/MNR_IJCAI25/dataset/datasets --gpu 0 --fp16 \
#                --image-size 80 --epochs 100 --seed 3407 --batch-size 32 --lr 0.001 --wd 1e-5 \
#                -a hpai_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                > ablation-main-hpai_rpv.log &


#python -u main.py --dataset-name RPV --dataset-dir /home/scxhc1/MNR_IJCAI25/dataset/datasets --gpu 0 --fp16 \
#                --image-size 80 --epochs 20 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                --resume /home/Chengtai_Li/桌面/MNR_IJCAI25/ckpts/save/RPV-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep200-seed3407-acc88_7/model_best.pth.tar \
#                > output_rpv.log &

