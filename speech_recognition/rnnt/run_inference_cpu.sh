

# tcmalloc:
export LD_PRELOAD=/home/mingfeim/packages/gperftools-2.8/install/lib/libtcmalloc.so

sockets=`lscpu | grep Socket | awk '{print $2}'`

root_dir=`pwd`
work_dir=$root_dir/mlperf-rnnt-librispeech
local_data_dir=$work_dir/local_data

scenario=Offline
if [[ "$1" == "--server" ]]; then
    scenario=Server
    shift
fi

batch_size=32
instances_per_socket=2
num_instances=`expr $sockets \* $instances_per_socket`

backend=pytorch
accuracy="" ### or "--accuracy"

log_dir=${work_dir}/${scenario}_${backend}
if [ ! -z ${accuracy} ]; then
    log_dir+=_accuracy
fi
log_dir+=rerun

python run.py --dataset_dir $local_data_dir \
    --manifest $local_data_dir/dev-clean-wav.json \
    --pytorch_config_toml pytorch/configs/rnnt.toml \
    --pytorch_checkpoint $work_dir/rnnt.pt \
    --scenario ${scenario} \
    --backend ${backend} \
    --log_dir ${log_dir} \
    --offline_batch_size ${batch_size} \
    --num_instances $num_instances \
    ${accuracy}

