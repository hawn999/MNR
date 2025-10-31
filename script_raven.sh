# nohup python -u main.py --dataset-name I-RAVEN --dataset-dir /home/Chengtai_Li/文档/AVR-PredRNet-main/datasets --gpu 0 --fp16 \
#                --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                > output_mnr.txt &> output_predrnet_iraven.log

# nohup python -u main.py --dataset-name I-RAVEN --dataset-dir /home/Chengtai_Li/文档/AVR-PredRNet-main/datasets --gpu 0 --fp16 \
#                --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                --resume /media/Chengtai_Li/hhee/CVPR25/RVN-OOD/Pred/IRVN/I-RAVEN-pred-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed3407/model_best.pth.tar \
#                > output_mnr.txt &> output_predrnet_iraven.log

# nohup python -u main.py --dataset-name RAVEN --dataset-dir /home/Chengtai_Li/文档/AVR-PredRNet-main/datasets --gpu 0 --fp16 \
#                --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                --resume /media/Chengtai_Li/hhee/CVPR25/RVN-OOD/Pred/RVN/RAVEN-pred-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed3407/model_best.pth.tar \
#                > output_mnr.txt &> output_predrnet_raven.log

# nohup python -u main.py --dataset-name RAVEN --dataset-dir /home/Chengtai_Li/文档/AVR-PredRNet-main/datasets --gpu 0 --fp16 \
#                --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                > output_mnr.txt &> output_predrnet_raven.log

# nohup python -u main.py --dataset-name RAVEN-FAIR --dataset-dir /home/Chengtai_Li/文档/AVR-PredRNet-main/datasets --gpu 0 --fp16 \
#                --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                --resume /media/Chengtai_Li/hhee/CVPR25/RVN-OOD/Pred/RVNF/RAVEN-FAIR-pred-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed3407/model_best.pth.tar \
#                > output_mnr.txt &> output_predrnet_raven-fair.log


nohup python -u main.py --dataset-name RAVEN-FAIR --dataset-dir /home/Chengtai_Li/文档/AVR-PredRNet-main/datasets --gpu 0 --fp16 \
               --image-size 80 --epochs 20 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
               -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
               --classifier-hidreduce 4 --show-detail --ckpt ckpts/ -e \
               --resume /home/Chengtai_Li/桌面/MNR_IJCAI25/ckpts/save/99RAVEN-FAIR-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep20-seed3407/model_best.pth.tar \
               > result_predrnet_raven-fair.log &

# nohup python -u main.py --dataset-name RAVEN-FAIR --dataset-dir /home/Chengtai_Li/文档/AVR-PredRNet-main/datasets --gpu 0 --fp16 \
#                --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                > output_mnr.txt &> output_predrnet_raven-fair.log

# nohup python -u main.py --dataset-name RAVEN-FAIR --dataset-dir /home/Chengtai_Li/文档/AVR-PredRNet-main/datasets --gpu 0 --fp16 \
#                --image-size 80 --epochs 20 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                --resume ckpts/load-RAVEN-FAIR-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep200-seed3407/model_best.pth.tar \
#                > output_mnr.txt &> alpha0.7-output_predrnet_raven-fair.log


# nohup python -u main.py --dataset-name I-RAVEN --dataset-dir /home/Chengtai_Li/文档/AVR-PredRNet-main/datasets --gpu 0 --fp16 \
#                --image-size 80 --epochs 200 --seed 3407 --batch-size 64 --lr 0.001 --wd 1e-5 \
#                -a hcv_pric_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                > output_hcv_pric_i-raven.log &


# nohup python -u main.py --dataset-name I-RAVEN --dataset-dir /home/Chengtai_Li/文档/AVR-PredRNet-main/datasets --gpu 0 --fp16 \
#                --image-size 80 --epochs 200 --seed 3407 --batch-size 64 --lr 0.001 --wd 1e-5 \
#                -a mrnet_pric_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --ckpt ckpts/ \
#                > output_mrnet_pric_i-raven.log &

# nohup python -u main.py --dataset-name I-RAVEN --dataset-dir /home/Chengtai_Li/文档/AVR-PredRNet-main/datasets --gpu 0 --fp16 \
#                --image-size 80 --epochs 1 --seed 3407 --batch-size 64 --lr 0.001 --wd 1e-5 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --ckpt ckpts/ --show-detail -e \
#                --resume ckpts/save/alpha0.7-I-RAVEN-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep100-seed3407/model_best.pth.tar \
#                > result-on-i-raven.log &