# nohup python -u main.py --dataset-name Unicode-N --dataset-dir /home/Chengtai_Li/桌面/MNR_IJCAI25/dataset/ --gpu 0 --fp16 \
#                --image-size 80 --epochs 200 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_analogy --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --show-detail --ckpt ckpts/ \
#                > output_unicode.log &




# nohup python -u main.py --dataset-name Unicode-N --dataset-dir /home/Chengtai_Li/桌面/MNR_IJCAI25/dataset/ --gpu 0 --fp16 \
#                --image-size 80 --epochs 20 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_analogy --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --show-detail --ckpt ckpts/ \
#                --resume /home/Chengtai_Li/桌面/MNR_IJCAI25/ckpts/save/Unicode-N-predrnet_analogy-prb3-b0.1c0.1-imsz80-wd1e-05-ep200-seed12345-acc70/model_best.pth.tar \
#                > output_unicode.log &

# nohup python -u main.py --dataset-name Unicode-N --dataset-dir /home/Chengtai_Li/桌面/MNR_IJCAI25/dataset/ --gpu 0 --fp16 \
#                --image-size 80 --epochs 20 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_analogy --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --show-detail --ckpt ckpts/ -e \
#                --resume /home/Chengtai_Li/桌面/MNR_IJCAI25/ckpts/Unicode-N-predrnet_analogy-prb3-b0.1c0.1-imsz80-wd1e-05-ep20-seed12345/checkpoint.pth.tar \
#                > output_unicode.log &

# nohup python -u main.py --dataset-name Unicode-N --dataset-dir /home/Chengtai_Li/桌面/MNR_IJCAI25/dataset/ --gpu 0 --fp16 \
#                --image-size 80 --epochs 20 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_analogy --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --show-detail --ckpt ckpts/ -e \
#                --resume /home/Chengtai_Li/桌面/MNR_IJCAI25/ckpts/Unicode-N-predrnet_analogy-prb3-b0.1c0.1-imsz80-wd1e-05-ep20-seed12345/model_best.pth.tar \
#                > output_unicode.log &



# ablation of main
# nohup python -u main.py --dataset-name Unicode-N --dataset-dir /home/Chengtai_Li/桌面/MNR_IJCAI25/dataset/ --gpu 0 --fp16 \
#                --image-size 80 --epochs 200 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a predrnet_analogy --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --show-detail --ckpt ckpts/ \
#                > ablation-main-uni-woLoss.log &

# nohup python -u main.py --dataset-name Unicode-N --dataset-dir /home/Chengtai_Li/桌面/MNR_IJCAI25/dataset/ --gpu 0 --fp16 \
#                --image-size 80 --epochs 100 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a mrnet_pric_analogy --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --show-detail --ckpt ckpts/ \
#                > ablation-mrnet_pric_unicode.log &

# nohup python -u main.py --dataset-name Unicode-N --dataset-dir /home/Chengtai_Li/桌面/MNR_IJCAI25/dataset/ --gpu 0 --fp16 \
#                --image-size 80 --epochs 500 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a hcv_pric_analogy --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --show-detail --ckpt ckpts/ \
#                > ablation-hcv_pric_unicode500.log &

# nohup python -u main.py --dataset-name Unicode-N --dataset-dir /home/Chengtai_Li/桌面/MNR_IJCAI25/dataset/ --gpu 0 --fp16 \
#                --image-size 80 --epochs 200 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
#                -a hpai_pric_analogy --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --show-detail --ckpt ckpts/ \
#                > ablation-hpai_pric-unicode-epoch200.log &

nohup python -u main.py --dataset-name Unicode-N --dataset-dir /home/Chengtai_Li/桌面/MNR_IJCAI25/dataset/ --gpu 0 --fp16 \
               --image-size 80 --epochs 500 --seed 2025 --batch-size 64 --lr 0.001 --wd 1e-5 \
               -a hcv_pric_v2_analogy --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
               --classifier-hidreduce 4 --show-detail --ckpt ckpts/ \
               > ablation-hcv_pric_v2-unicode500_2025.log &

# nohup python -u main.py --dataset-name Unicode-N --dataset-dir /home/Chengtai_Li/桌面/MNR_IJCAI25/dataset/ --gpu 0 --fp16 \
#                --image-size 80 --epochs 100 --seed 12345 --batch-size 64 --lr 0.001 --wd 1e-5 \
#                -a darr_analogy --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#                --classifier-hidreduce 4 --show-detail --ckpt ckpts/ \
#                > ablation-darr_analogy-unicode.log &