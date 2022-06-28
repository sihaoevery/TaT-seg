#For MobileNet, Change the batch size (BS) to 12.

GPU_IDS=0 EPOCH=100 BS=16 Scheduler=poly LR=0.007 Crop=513

CUDA_VISIBLE_DEVICES=$GPU_IDS python train_with_distillation_reviewkd.py \
--tnet pretrained/YOUR_resnet101_cocostuff10k.pth.tar \
--backbone resnet18 \
--gpu-ids $GPU_IDS \
--epochs $EPOCH \
--lr $LR \
--lr-scheduler $Scheduler \
--crop-size $Crop \
--dataset cocostuff10k \
--nesterov \
--batch-size $BS \
--flag epoch:{$EPOCH}_{$Scheduler}_lr:{$LR}_bs:{$BS}_crop:{$Crop}_reviewkd

