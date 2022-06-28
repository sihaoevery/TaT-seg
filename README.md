# Knowledge Distillation via the Target-aware Transformer (CVPR2022)
Experiments on semantic segmentation of our work. See this [link](https://github.com/sihaoevery/TaT) for experiments on ImageNet.

## Requirement
- python 3.8
- pytorch >= 1.9.0
- torchvision 0.11.1
- einops

### Note
 All the experiments are conducted on a single Nvidia A100 (40GB). Multi-gpu environment hasn't been tested.

## Overview
### Before getting started
Please modify the dataset path on the file [mypath.py](./mypath.py) according to your system.

### Implementation
Our model is implemented on [./distiller_tat](./distiller_tat.py). 

We also provide the implementation of ReveiwKD on [./distiller_reveiwkd](./distiller_reviewkd.py) and other methods (KD/FitNet/AT/ICKD) on [./distiller_comp](./distiller_comp.py).

### Execution
The executable file is [./train_with_distillation_tat](./train_with_distillation_tat.py). 

## Training a teacher model
ResNet101 is used as teacher backbone. 
### Pascal VOC 
We use the official [model](https://github.com/jfzhang95/pytorch-deeplab-xception). Please download the checkpoint from [here](https://drive.google.com/file/d/1Pz2OT5KoSNvU5rc3w5d2R8_0OBkKSkLR/view) and put it on [./pretrained/](./pretrained/) .

### COCOStuff-10k
We train the teacher on our own. Simply running:
```bash
sh ./train_cocostuff10k_baseline.sh
```
## Training with distillation
Please refer to the shell scripts. For instance, distilling the ResNet101 to ResNet18 on Pascal VOC:
```bash
sh ./train_voc_resnet18.sh
```


## TO-DO
- [ ] Upload pre-trained cocostuff-10k teacher model
- [ ] Upload training log
- [ ] Dataset preparation
