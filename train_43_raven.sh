## 43RAVEN
python -u main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/MNR_IJCAI25/dataset/datasets --gpu 0,1 --fp16 \
            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
            -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
            --classifier-hidreduce 4 --ckpt ckpts/4GIPRB_3HYLA_RAVEN_retrain_ \
            --reduce-planes 128 --num-hylas 3 --workers 2\
            # --resume /home/scxhc1/MNR_IJCAI25/ckpts/4GIPRB_3HYLA_RAVEN/checkpoint.pth.tar\
            2>&1 | tee log/4GIPRB_3HYLA_RAVEN_retrain_hyla_000.txt

#python main.py \
#  --dataset-dir //home/scxhc1/nvme_data/resized_datasets_raven \
#  --image-size 80 --seed 3407 --batch-size 64 --epochs 10\
#  --dataset-name RAVEN \
#  -a predrnet_raven\
#  --reduce-planes 128 --num-hylas 3 --workers 2\
#  --resume /home/scxhc1/MNR_IJCAI25/ckpts/4GIPRB_3HYLA_RAVEN/model_best.pth.tar \
#  --evaluate \
#  --tsne \
#  --gpu 0,1



