# rpv ablation li
path="/home/scxhc1/MNR_IJCAI25/dataset/datasets"
bz=16
# path="/home/scxdx2/nvme_data/MNR_data/datasets"

PYTHONUNBUFFERED=1 python main.py --dataset-name RPV --dataset-dir $path --gpu 0,1 --fp16 \
                --image-size 80 --epochs 100 --seed 3407 --batch-size $bz --lr 0.001 --wd 1e-5 \
                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
                --classifier-hidreduce 4 --ckpt ckpts/43 \
                --reduce-planes 128 --num-hylas 3 --workers 2\
                2>&1 | tee rpv_log/43.txt
PYTHONUNBUFFERED=1 python main.py --dataset-name RPV --dataset-dir $path --gpu 0,1 --fp16 \
                --image-size 80 --epochs 100 --seed 3407 --batch-size $bz --lr 0.001 --wd 1e-5 \
                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
                --classifier-hidreduce 4 --ckpt ckpts/44 \
                --reduce-planes 128 --num-hylas 4 --workers 4\
                2>&1 | tee rpv_log/44.txt
PYTHONUNBUFFERED=1 python main.py --dataset-name RPV --dataset-dir $path --gpu 0,1 --fp16 \
                --image-size 80 --epochs 100 --seed 3407 --batch-size $bz --lr 0.001 --wd 1e-5 \
                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
                --classifier-hidreduce 4 --ckpt ckpts/41_no_conv \
                --reduce-planes 128 --num-hylas 1 --workers 4\
                2>&1 | tee rpv_log/41.txt
PYTHONUNBUFFERED=1 python main.py --dataset-name RPV --dataset-dir $path --gpu 0,1 --fp16 \
                --image-size 80 --epochs 100 --seed 3407 --batch-size $bz --lr 0.001 --wd 1e-5 \
                -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
                --classifier-hidreduce 4 --ckpt ckpts/42_no_conv \
                --reduce-planes 128 --num-hylas 2 --workers 4\
                2>&1 | tee rpv_log/42.txt


 PYTHONUNBUFFERED=1 python main.py --dataset-name RPV --dataset-dir $path --gpu 0,1 --fp16 \
                 --image-size 80 --epochs 100 --seed 3407 --batch-size $bz --lr 0.001 --wd 1e-5 \
                 -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
                 --classifier-hidreduce 4 --ckpt ckpts/53 \
                 --reduce-planes 160 --num-hylas 3 --workers 4\
                 2>&1 | tee rpv_log/53.txt

 PYTHONUNBUFFERED=1 python main.py --dataset-name RPV --dataset-dir $path --gpu 0,1 --fp16 \
                 --image-size 80 --epochs 100 --seed 3407 --batch-size $bz --lr 0.001 --wd 1e-5 \
                 -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
                 --classifier-hidreduce 4 --ckpt ckpts/33 \
                 --reduce-planes 96 --num-hylas 3 --workers 4\
                 2>&1 | tee rpv_log/33.txt

 PYTHONUNBUFFERED=1 python main.py --dataset-name RPV --dataset-dir $path --gpu 0,1 --fp16 \
                 --image-size 80 --epochs 100 --seed 3407 --batch-size $bz --lr 0.001 --wd 1e-5 \
                 -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
                 --classifier-hidreduce 4 --ckpt ckpts/23 \
                 --reduce-planes 128 --num-hylas 3 --workers 4\
                 2>&1 | tee rpv_log/23.txt


 PYTHONUNBUFFERED=1 python main.py --dataset-name RPV --dataset-dir $path --gpu 0,1 --fp16 \
                 --image-size 80 --epochs 100 --seed 3407 --batch-size $bz --lr 0.001 --wd 1e-5 \
                 -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
                 --classifier-hidreduce 4 --ckpt ckpts/13 \
                 --reduce-planes 32 --num-hylas 3 --workers 4\
                 2>&1 | tee rpv_log/13.txt

# PYTHONUNBUFFERED=1 python main.py --dataset-name RPV --dataset-dir $path --gpu 0,1 --fp16 \
#                 --image-size 80 --epochs 100 --seed 3407 --batch-size 64 --lr 0.001 --wd 1e-5 \
#                 -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
#                 --classifier-hidreduce 4 --ckpt ckpts/44 \
#                 --reduce-planes 128 --num-hylas 4 --workers 4\
#                 2>&1 | tee 44_re.txt
# PYTHONUNBUFFERED=1 python main.py --dataset-name RPV --dataset-dir $path --gpu 0 --fp16 \
#                 --image-size 80 --epochs 100 --seed 3407 --batch-size 64 --lr 0.001 --wd 1e-5 \
#                 -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
#                 --classifier-hidreduce 4 --ckpt ckpts/41 \
#                 --reduce-planes 128 --num-hylas 1 --workers 4\
#                 2>&1 | tee 41.txt
# PYTHONUNBUFFERED=1 python main.py --dataset-name RPV --dataset-dir $path --gpu 0 --fp16 \
#                 --image-size 80 --epochs 100 --seed 3407 --batch-size 64 --lr 0.001 --wd 1e-5 \
#                 -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 --in-channels 3 \
#                 --classifier-hidreduce 4 --ckpt ckpts/42 \
#                 --reduce-planes 128 --num-hylas 2 --workers 4\
#                 2>&1 | tee 42.txt



