# 20717347 MYP Code - Hierarchical Abstract Visual Reasoning via Multi-Perspective Rule Induction and Dynamic Composition

This repository contains the code implementation for the MYP (Master's Year Project) associated with *Hierarchical Abstract Visual Reasoning via Multi-Perspective Rule Induction and Dynamic Composition* by 20717347(scxhc1).


## Important Notice

As the paper has not yet been published, the network model contents are not uploaded at this time. If you require the full code or have any other inquiries related to the project, please contact me via email:  
[scxhc1@nottingham.edu.cn](mailto:scxhc1@nottingham.edu.cn)

## Code environments and toolkits

- OS: Ubuntu 18.04.5
- CUDA: 12.6
- Python: 3.10.18
- Toolkit: PyTorch 2.7.0+cu126
- 2x GPU: NVIDIA RTX A5000
- [thop](https://github.com/Lyken17/pytorch-OpCounter)
- [muon](https://github.com/KellerJordan/Muon)

### Experiments

#### Dataset Structure

Please prepare datasets with the following structure:


```markdown
your_dataset_root_dir/

    ├─I-RAVEN (RAVEN or RAVEN-FAIR)
    │  ├─center_single
    │  ├─distribute_four
    │  ├─distribute_nine
    │  ├─in_center_single_out_center_single
    │  ├─in_distribute_four_out_center_single
    │  ├─left_center_single_right_center_single
    │  └─up_center_single_down_center_single

```

#### Training and Evaluation
You can train different variants of the PERIC model by specifying the `--reduce-planes` argument (K*32) and `--num-hylas` argument (L). Additionally, you can adjust other training parameters to customize your training process. Examples:

**Train PERIC with K=4 and L=3 on RAVEN:**

```python
python main.py --dataset-name RAVEN --dataset-dir your_dataset_root_dir --gpu 0,1 --fp16 \
               --image-size 80 --epochs 200 --seed 12345 --batch-size 108 --lr 0.001 --wd 1e-5 \
               -a model_name --block-drop 0.1 --classifier-drop 0.1 \
               --ckpt your_checkpoint_dir \
               --reduce-planes 128 --num-hylas 3\
```
**Evaluation:**
To evaluate a trained model on the test set, use the `-e` (or `--evaluate`) flag along with `--resume` to specify the checkpoint path. Use `--show-detail` to print detailed accuracy.
```python
python main.py --dataset-name RAVEN --dataset-dir your_dataset_root_dir --gpu 0,1 --fp16 \
               --image-size 80 -a model_name \
               -e --resume your_checkpoint_dir/model_best.pth.tar \
               --show-detail

## Special Thanks

We would like to express our gratitude to the following projects and contributors:

* **Framework Base**: This code is built upon the framework provided by [AVR-PredRNet-and-SSPredRNet](https://github.com/ZjjConan/AVR-PredRNet-and-SSPredRNet/tree/main). We thank the authors for their open-source contribution.
