## prb
now=$(date +"%Y-%m-%d-%H-%M-%S")
python -u cot_main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/i-raven-cot/dataset_test_3 --gpu 0,1 --fp16 \
            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
            -a cot_prb_mask3 --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
            --classifier-hidreduce 4 --ckpt ckpts/cot_prb_ir_adam_${now}\
            --workers 2 --in-channels 1\
            2>&1 | tee log/cot/${now}_cotv3_prb_mask3.txt

# now=$(date +"%Y-%m-%d-%H-%M-%S")
# python -u cot_main.py --dataset-name RAVEN-FAIR --dataset-dir /home/scxhc1/nvme_data/datasets/ --gpu 0,1 --fp16 \
#             --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#             -a raven_mask12 --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#             --classifier-hidreduce 4 --ckpt ckpts/cot_prb_ir_adam_${now}\
#             --workers 2 --in-channels 1\
#             2>&1 | tee log/cot/${now}_ravenf_prb_mask12.txt

#now=$(date +"%Y-%m-%d-%H-%M-%S")
#python -u cot_main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/i-raven-cot/dataset_test_3 --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a cot_prb_mask34 --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/cot_prb_ir_adam_${now}\
#            --workers 2 --in-channels 1\
#            2>&1 | tee log/cot/${now}_cotv3_prb_mask34.txt
#
#now=$(date +"%Y-%m-%d-%H-%M-%S")
#python -u cot_main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/i-raven-cot/dataset_test_3 --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a cot_prb --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/cot_prb_ir_adam_${now}\
#            --workers 2 --in-channels 1\
#            2>&1 | tee log/cot/${now}_cotv3_prb_nomask.txt
#now=$(date +"%Y-%m-%d-%H-%M-%S")
#python -u cot_main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/i-raven-cot/dataset_test_3 --gpu 0,1 --fp16 \
#            --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#            -a cot_prb_mask4 --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#            --classifier-hidreduce 4 --ckpt ckpts/cot_prb_ir_adam_${now}\
#            --workers 2 --in-channels 1\
#            2>&1 | tee log/cot/${now}_cotv3_prb_mask4.txt

# now=$(date +"%Y-%m-%d-%H-%M-%S")
# python -u cot_main.py --dataset-name I-RAVEN --dataset-dir /home/scxhc1/i-raven-cot/dataset_test_3 --gpu 0,1 --fp16 \
#             --image-size 80 --epochs 200 --seed 3407 --batch-size 128 --lr 0.001 --wd 1e-5 \
#             -a cot_prb_mask234 --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
#             --classifier-hidreduce 4 --ckpt ckpts/cot_prb_ir_adam_${now}\
#             --workers 2 --in-channels 1\
#             2>&1 | tee log/cot/${now}_cotv3_prb_mask234.txt










