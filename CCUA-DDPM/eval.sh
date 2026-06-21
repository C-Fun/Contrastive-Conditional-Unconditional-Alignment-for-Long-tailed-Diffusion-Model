export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

DIR=$0
LOGDIR=$1
OMEGA=$2


PYTHONPATH="$(dirname $DIR)/":"$(dirname $DIR)/DDPM/":$PYTHONPATH \
python  ./DDPM/evaluate.py --flagfile $LOGDIR/flagfile.txt \
            --notrain --eval --parallel \
            --conditional --cfg \
            --logdir $LOGDIR \
            --ckpt_step 100000 \
            --num_images 10000 --batch_size 2048 \
            --sample_method "ddim" \
            --ddim_skip_step 10 \
            --omega $OMEGA --metrics "fid,kid,fid_tail" --seed 42