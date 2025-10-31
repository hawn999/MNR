nohup python -u main.py --dataset-name PGM --dataset-dir ./PGM/neutral/ --gpu 0 --fp16 \
               --image-size 80 --epochs 100 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
               -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
               --classifier-hidreduce 4 --ckpt ckpts/ \
               --resume weights_PGM/neutral-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-07-ep100-seed12345/checkpoint.pth.tar \
               > output_pgm_neutral.txt &