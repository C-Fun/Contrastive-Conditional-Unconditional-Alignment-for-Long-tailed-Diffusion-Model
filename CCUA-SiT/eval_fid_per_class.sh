DIR=$0
PROJECT=$1
MODEL_NAME=$2
MASTER_PORT=$3
CKPT_NAME=$4
N_GPUS=$5
SELECTED_CLASSES=$6
CFG_SCALE=7.5
SAMPLE_BS=4

LATEST_FOLDER=$(ls -d results/$PROJECT/[0-9][0-9][0-9]-${MODEL_NAME}-Linear-velocity-None 2>/dev/null | sort -V | tail -n 1)
#LATEST_CKPT=$(ls $LATEST_FOLDER/checkpoints/*.pt 2>/dev/null | sort -V | tail -n 1)
LATEST_CKPT=$LATEST_FOLDER/checkpoints/$CKPT_NAME

echo "latest folder: $LATEST_FOLDER"
echo "latest ckpt: $LATEST_CKPT"

# The .npz file is named after the folder created by sample_ddp.py
CKPT_NAME=$(basename "$LATEST_CKPT" .pt)
SAMPLE_NPZ="samples/${PROJECT}_${SELECTED_CLASSES}/${MODEL_NAME}-${CKPT_NAME}-cfg-${CFG_SCALE}-${SAMPLE_BS}-ODE-50-dopri5.npz"
EVAL_TXT="samples/${PROJECT}_${SELECTED_CLASSES}/${MODEL_NAME}-${CKPT_NAME}-cfg-${CFG_SCALE}-${SAMPLE_BS}-ODE-50-dopri5.txt"

torchrun --nnodes=1 --nproc_per_node=$N_GPUS --master-port $MASTER_PORT ./SiT/sample_ddp_per_class_fid.py ODE \
         --data-path /mnt/data2/imagenet2012/train/ \
         --image-size 256 --num-sampling-steps 50 --cfg-scale $CFG_SCALE --per-proc-batch-size $SAMPLE_BS \
         --model SiT-S/2 --num-fid-samples 20000 --ckpt "$LATEST_CKPT" --sample-dir "samples/${PROJECT}_${SELECTED_CLASSES}/" \
         --selected-classes $SELECTED_CLASSES | tee -a "$EVAL_TXT"

#python evaluator.py ./VIRTUAL_imagenet256_labeled.npz "$SAMPLE_NPZ" | tee -a "$EVAL_TXT"