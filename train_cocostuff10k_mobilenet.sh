GPU_IDS=0 EPOCH=100 STEP=30 LR=0.007 ALPHA=0.4 BETA=1.0 GAMMA=1.0
N=16 M=16 HEAD=64 Anchor_H=32 Anchor_W=32 BS=12 ATTN=stack_attn

CUDA_VISIBLE_DEVICES=$GPU_IDS python train_with_distillation_tat.py \
--tnet pretrained/YOUR_resnet101_cocostuff10k.pth.tar \
--backbone mobilenet \
--gpu-ids $GPU_IDS \
--dataset cocostuff10k \
--use-sbd \
--nesterov \
--epochs $EPOCH \
--batch-size $BS \
--heads $HEAD \
--lr $LR \
--lr-scheduler cos \
--lr-step $STEP \
--alpha $ALPHA \
--beta $BETA \
--gamma $GAMMA \
-n $N \
-m $M \
--anchor_h $Anchor_H \
--anchor_w $Anchor_W \
--attn-type $ATTN \
--flag epoch:{$EPOCH}_cos_lr:{$LR}_bs:{$BS}_p:{$N}_anc:{$Anchor_H}_h:{$HEAD}_a:{$ALPHA}_b:{$BETA}_g:{$GAMMA}_attn:{$ATTN}