# STViT-R
This is an official implementation for "Making Vision Transformers Efficient from A Token Sparsification View". It is based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer).

**Notes:**
We will further clean the code and release the checkpoints in the future.

> **Object Detection and Instance Segmentation**: See [STViT-R-Object-Detection](https://github.com/changsn/STViT-R-Object-Detection)


## Results on ImageNet
| Model | Accuracy | Log |
| :---: | :---: | :---: |
| STViT-R-Swin-S | 82.7 | [Link](https://github.com/changsn/STViT-R/blob/main/log/small.txt) |
| STViT-R-Swin-B | 83.2 | [Link](https://github.com/changsn/STViT-R/blob/main/log/base.txt) |

## Usage
### Installation
See [get_started.md](get_started.md) for a quick installation.

### Training
```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \ 
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```
For example, to train `Swin Transformer` with 8 GPU on a single node for 300 epochs, run:

`STViT-R-Swin-S`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin_small_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128 
```

`STViT-R-Swin-B`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin_base_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 64 \
--accumulation-steps 2 [--use-checkpoint]
```

## Citing STViT-R
```
@inproceedings{chang2023making,
  title={Making Vision Transformers Efficient from A Token Sparsification View},
  author={Chang, Shuning and Wang, Pichao and Lin, Ming and Wang, Fan and Zhang, David Junhao and Jin, Rong and Shou, Mike Zheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6195--6205},
  year={2023}
}
```
