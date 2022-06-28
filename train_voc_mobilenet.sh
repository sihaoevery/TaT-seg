GPU_IDS=0 EPOCH=100 STEP=30 LR=0.004 ALPHA=0.1 BETA=1.0 GAMMA=0.5
N=32 M=32 HEAD=1024 Anchor_H=64 Anchor_W=64 BS=12 ATTN=stack_attn

CUDA_VISIBLE_DEVICES=$GPU_IDS python train_with_distillation_tat.py \
--backbone mobilenet \
--gpu-ids $GPU_IDS \
--dataset pascal \
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