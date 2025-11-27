now=$(date +"%Y-%m-%d-%H-%M-%S")
python -u main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/nvme_data/datasets/ --gpu 0,1 --fp16 \
            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
            -a vit --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
            --classifier-hidreduce 4 --ckpt ckpts/${now}_v3_\
            --workers 2 --in-channels 1\
            2>&1 | tee log/${now}_vit_r.txt
# --resume ./ckpts/2025-11-22-16-11-53_v3_RAVEN-vit-ext3-b0.1c0.1-imsz80-wd1e-05-ep200-seed3407/checkpoint.pth.tar \
            