GPU_IDS=0 LR=0.004 BS=16 Crop=513 Scheduler=poly EPOCH=200

CUDA_VISIBLE_DEVICES=$GPU_IDS python train.py \
--nesterov \
--gpu-ids $GPU_IDS \
--backbone resnet101 \
--batch-size $BS \
--crop-size $Crop \
--workers 4 \
--epochs $EPOCH \
--lr $LR \
--lr-scheduler $Scheduler \
--dataset cocostuff10k \
--flag epoch:{$EPOCH}_{$Scheduler}_lr:{$LR}_bs:{$BS}_crop:{$Crop}_baseline



