

# tcmalloc:
export LD_PRELOAD=/home/mingfeim/packages/gperftools-2.8/install/lib/libtcmalloc.so

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
LAST_CORE=`expr $CORES - 1`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"

export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "\n### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"

### single socket test
echo -e "\n### using OMP_NUM_THREADS=$CORES"
PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"
echo -e "### using $PREFIX\n"
OMP_NUM_THREADS=$CORES $PREFIX ./run.sh
