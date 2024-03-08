#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/megadepth_test_1500.py"
main_cfg_path="configs/loftr/outdoor/buggy_pos_enc/loftr_ds.py"
#ckpt_path="/root/LoFTR/logs/tb_logs/outdoor-ds-420-bs=4/version_1/checkpoints/epoch=0-auc@5=0.369-auc@10=0.538-auc@20=0.680.ckpt"
#ckpt_path="/root/LoFTR/logs/tb_logs/outdoor-ds-420-bs=4/version_4/checkpoints/epoch=0-auc@5=0.370-auc@10=0.537-auc@20=0.681.ckpt"
ckpt_path="weights/outdoor_quad.ckpt"
dump_dir="dump/loftr_ds_outdoor_480"
profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=4
torch_num_workers=4
batch_size=1  # per gpu

source /root/miniconda3/etc/profile.d/conda.sh
conda activate loftr
python -u ./test.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark 
    