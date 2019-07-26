script=$1
id=$2
#source ~/anaconda3/bin/activate nmt_env
interpolated_nmt=/home/pham/One_Model
MYPYTHON=~/anaconda3/envs/nmt_env/bin/
MYPYLIB=~/anaconda3/envs/nmt_env/lib/
export PATH=$MYPYTHON:${PATH}
export PYTHONPATH=$interpolated_nmt:$MYPYLIB:$MYPYLIB/python2.7:$MYPYLIB/python2.7/dists-packages
#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64/:$HOME/cuda/lib64
#export CPATH=/home/shared/lib/cuDNNv5/include:$CPATH
export LIBRARY_PATH=/home/shared/lib/cuDNNv5/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=$id
which python
python -c 'import tensorflow; print "tensorflow OK"; import interpolated_nmt'

python -u $script
