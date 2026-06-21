export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

DIR=$0

PYTHONPATH="$(dirname $DIR)/":"$(dirname $DIR)/DDPM/":$PYTHONPATH \
python ./DDPM/main.py \
    --train --parallel \
    --data_type tinyimgnetlt --data_path /mnt/data2/tiny-imagenet-200/ \
    --img_size 64 --num_class 200 --imb_factor 0.01 \
    --logdir ./logs/ccua/tinyimgnetlt_imb0_01/ \
    --total_steps 100001 --batch_size 128 \
    --cfg --conditional \
    --notransfer_x0 \
    --nocbdm \
    --ccua_ucl 1.0 --ccua_al 1.0 --brs --brs_factor 0.1

