export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export WANDB_KEY="bca43cb3397ab6b7f7167e413fb925a79c9239f1"
export ENTITY="ucmerced-fangchen"

METHOD="ccua-N_4-brs0_1-ucl0_05-al0_05"
EPOCH=160
MACHINE_IP="a6000ada"
BS=48
N_GPUS=4
MASTER_PORT=2300
CKPT_NAME=0900000.pt
#SELECTED_CLASSES="tail"

export PROJECT="eccv-${METHOD}-imb0_01-epoch${EPOCH}-${MACHINE_IP}-4gpus-bs${BS}"

export PYTHONPATH="./SiT/:$PYTHONPATH"
torchrun --nnodes=1 --nproc_per_node=$N_GPUS --master-port $MASTER_PORT ./SiT/train.py \
         --model SiT-S/2 --data-path /mnt/data2/imagenet2012/ --imb-factor 0.01 --brs \
         --epochs $EPOCH --global-batch-size $BS --wandb --sample-every 10000 --results-dir "results/$PROJECT/" \
         --ccua-ucl 0.05 --ccua-al 0.05 --brs-f 0.1 --ccua-act-layer "N/4" --ccua-mode "parallel"

bash sample_eval_pipe.sh $PROJECT "SiT-S-2" $MASTER_PORT $CKPT_NAME $N_GPUS

#bash eval_fid_per_class.sh $PROJECT "SiT-S-2" $MASTER_PORT $CKPT_NAME $N_GPUS $SELECTED_CLASSES
