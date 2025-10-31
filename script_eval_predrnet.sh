
python main.py --dataset-name RAVEN --dataset-dir /home/scxhc1/nvme_data/resized_datasets_raven --gpu 0,1 \
               --image-size 80 -a predrnet_raven \
               -e --resume /home/scxhc1/MNR_IJCAI25/ckpts/4GIPRB_3HYLA_no_resume_youRAVEN-predrnet_raven-prb3-b0.1c0.1-imsz80-wd1e-05-ep200-seed3407/model_best.pth.tar \
               --show-detail