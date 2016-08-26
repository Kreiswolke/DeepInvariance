import sys
sys.path.append('/mnt/antares_raid/home/oliver/nideep')
sys.path.append('/mnt/antares_raid/home/oliver/Scripts/autoencoder')
from nideep.eval.inference import infer_to_h5_fixed_dims, infer_to_lmdb
from utils import set_up_dir
import caffe

def store_layerwise_activations(net_prototxt, model, phase, keys, n, dst_fpath):
    '''
    phase = caffe.TRAIN
    proto = '/mnt/antares_raid/home/oliver/Scripts/autoencoder_v2/autoencoder_with_MLP/autoencoder_with_MLP_net.prototxt'
    model = '/mnt/antares_raid/home/oliver/Scripts/autoencoder_v2/autoencoder_with_MLP/snapshots_with_MLP/_iter_780000.caffemodel'
    lmdb_path = '/mnt/antares_raid/home/oliver/Scripts/autoencoder_v2/MNIST_lmdb/MNIST_TRAIN_60000_rot_lmdb/shuffled/'
    dst_fpath =  "/mnt/antares_raid/home/oliver/Scripts/autoencoder_v2/autoencoder_with_MLP/results/res_train.hdf5"
    n = 780
    '''
    set_up_dir("/".join(dst_fpath.split("/")[:-1])+"/")
    print("path created: {}".format("/".join(dst_fpath.split("/")[:-1])+"/"))
    print("Destination: {}".format(dst_fpath))

    print(net_prototxt, model, phase)
    net = caffe.Net(net_prototxt, model, phase)
    infer_to_h5_fixed_dims(net, keys, n, dst_fpath, preserve_batch=False)
    print('Done creating %s'%dst_fpath)
    
