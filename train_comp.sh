GPU_IDS=0 EPOCH=100 BS=16 Scheduler=poly LR=0.007 Crop=513 Comp=at
N=8 M=8
CUDA_VISIBLE_DEVICES=$GPU_IDS python train_with_distillation_comp.py \
--tnet pretrained/YOUR_resnet101_cocostuff10k.pth.tar \
--backbone resnet18 \
--gpu-ids $GPU_IDS \
--epochs $EPOCH \
--lr $LR \
--lr-scheduler $Scheduler \
--crop-size $Crop \
-n $N \
-m $M \
--dataset cocostuff10k \
--nesterov \
--comp $Comp \
--batch-size $BS \
--flag epoch:{$EPOCH}_p:{$N}_{$Scheduler}_lr:{$LR}_bs:{$BS}_crop:{$Crop}_{$Comp}
