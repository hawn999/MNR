now=$(date +"%Y-%m-%d-%H-%M-%S")
# prb
python -u main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/nvme_data/cot_test/v2_test1 --gpu 0,1 --fp16 \
            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
            -a cot_prb --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
            --classifier-hidreduce 4 --ckpt ckpts/cot_prb_ir_adam_${now}\
            --workers 2 --in-channels 1\
            2>&1 | tee log/${now}_cotv3_prb_ir_adam.txt
















