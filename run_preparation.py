import sys
sys.path.append('/mnt/antares_raid/home/oliver/nideep')
sys.path.append('/mnt/antares_raid/home/oliver/Scripts/autoencoder')
import os
from TransformMNIST import MNISTtransformer
from CorruptMNIST import MNISTcorrupter
import subprocess
import sys
import caffe
import matplotlib
import numpy as np
import lmdb
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import math
from nideep.eval.learning_curve import LearningCurve
from nideep.eval.inference import infer_to_h5_fixed_dims, infer_to_lmdb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from nideep.eval.learning_curve import LearningCurve
from nideep.eval.eval_utils import Phase
from utils import set_up_dir
from edit_prototxt import edit_prototxt, edit_solver


date = '2008/'
caffe_root = '/mnt/antares_raid/home/oliver/adhara/src/caffe/build/tools/caffe'
lmdb_root = '/mnt/raid/dnn/data_oliver/'
root =  "/mnt/antares_raid/home/oliver/Experiments/" + date
init_weights = 'TEST'

replacement_dict_solver = {
    'network': [], 
    'testiter': 130,
    'testinterval': 780,
    'maxiter': 390000,
    'snapshotprefix': []
}

#### ALL CASES
cases = {
         'R':{'template_net': "/mnt/antares_raid/home/oliver/Scripts/template_AE_net.prototxt" , 
              'template_solver': "/mnt/antares_raid/home/oliver/Scripts/template_autoencoder_solver.prototxt",
              'replacement_dict': 
                  { 
                  'train_net': '"{}"'.format(lmdb_root + "/lmdb/MNIST_TRAIN_60000_rot_lmdb/shuffled/"),
                  'test_net': '"{}"'.format(lmdb_root + "/lmdb/MNIST_TEST_10000_rot_lmdb/shuffled/" )
                  }
              },
                
         'UR': {'template_net': "/mnt/antares_raid/home/oliver/Scripts/template_AE_net_in_out.prototxt",
                'template_solver': "/mnt/antares_raid/home/oliver/Scripts/template_autoencoder_solver.prototxt",
                'replacement_dict': 
                    { 
                    'train_net': '"{}"'.format(lmdb_root + "/lmdb/MNIST_TRAIN_60000_rot_lmdb/shuffled/"),
                    'test_net': '"{}"'.format(lmdb_root + "/lmdb/MNIST_TEST_10000_rot_lmdb/shuffled/" ),
                    'train_net_out': '"{}"'.format(lmdb_root + "/lmdb/MNIST_TRAIN_60000_unrot_lmdb/shuffled/"),
                    'test_net_out': '"{}"'.format(lmdb_root + "/lmdb/MNIST_TEST_10000_unrot_lmdb/shuffled/") 
                    }
                },
    
         'N25NR':{'template_net': "/mnt/antares_raid/home/oliver/Scripts/template_AE_net_in_out.prototxt" , 
                  'template_solver': "/mnt/antares_raid/home/oliver/Scripts/template_autoencoder_solver.prototxt",
                  'replacement_dict': 
                   {
                    'train_net': '"{}"'.format(lmdb_root + "lmdb_corrupted_25/MNIST_TRAIN_60000_corrupt_px_196_rot_lmdb/shuffled/"),
                    'test_net': '"{}"'.format(lmdb_root + "lmdb_corrupted_25/MNIST_TEST_10000_corrupt_px_196_rot_lmdb/shuffled/" ),
                    'train_net_out': '"{}"'.format(lmdb_root +"lmdb_corrupted_25/MNIST_TRAIN_60000_corrupt_px_196_unrot_lmdb/shuffled/"),
                    'test_net_out': '"{}"'.format(lmdb_root + "lmdb_corrupted_25/MNIST_TEST_10000_corrupt_px_196_unrot_lmdb/shuffled/") 
                    }
                 },
    
                
         'N25NUR': {'template_net': "/mnt/antares_raid/home/oliver/Scripts/template_AE_net_in_out.prototxt" , 
                    'template_solver': "/mnt/antares_raid/home/oliver/Scripts/template_autoencoder_solver.prototxt",
                    'replacement_dict': 
                   {
                    'train_net': '"{}"'.format(lmdb_root + "lmdb_corrupted_25/MNIST_TRAIN_60000_corrupt_px_196_unrot_corrupted_lmdb/shuffled/"),
                    'test_net': '"{}"'.format(lmdb_root + "lmdb_corrupted_25/MNIST_TEST_10000_corrupt_px_196_unrot_corrupted_lmdb/shuffled/"),
                    'train_net_out': '"{}"'.format(lmdb_root +"lmdb_corrupted_25/MNIST_TRAIN_60000_corrupt_px_196_unrot_lmdb/shuffled/"),
                    'test_net_out': '"{}"'.format(lmdb_root + "lmdb_corrupted_25/MNIST_TEST_10000_corrupt_px_196_unrot_lmdb/shuffled/") 
                    }
                 },
                    
         'N50NR':{'template_net': "/mnt/antares_raid/home/oliver/Scripts/template_AE_net_in_out.prototxt" , 
                  'template_solver': "/mnt/antares_raid/home/oliver/Scripts/template_autoencoder_solver.prototxt",
                  'replacement_dict':  
                 {
                  'train_net': '"{}"'.format(lmdb_root + "lmdb_corrupted_50/MNIST_TRAIN_60000_corrupt_px_392_rot_lmdb/shuffled/"),
                  'test_net': '"{}"'.format(lmdb_root + "lmdb_corrupted_50/MNIST_TEST_10000_corrupt_px_392_rot_lmdb/shuffled/"),
                  'train_net_out': '"{}"'.format(lmdb_root + "lmdb_corrupted_50/MNIST_TRAIN_60000_corrupt_px_392_unrot_lmdb/shuffled/"),
                  'test_net_out': '"{}"'.format(lmdb_root + "lmdb_corrupted_50/MNIST_TEST_10000_corrupt_px_392_unrot_lmdb/shuffled/") 
                 }
                },

         'N50NUR': {'template_net': "/mnt/antares_raid/home/oliver/Scripts/template_AE_net_in_out.prototxt" , 
                  'template_solver': "/mnt/antares_raid/home/oliver/Scripts/template_autoencoder_solver.prototxt",
                  'replacement_dict':  
                 {
                  'train_net': '"{}"'.format(lmdb_root + "lmdb_corrupted_50/MNIST_TRAIN_60000_corrupt_px_392_unrot_corrupted_lmdb/shuffled/"),
                  'test_net': '"{}"'.format(lmdb_root + "lmdb_corrupted_50/MNIST_TEST_10000_corrupt_px_392_unrot_corrupted_lmdb/shuffled/" ),
                  'train_net_out': '"{}"'.format(lmdb_root + "lmdb_corrupted_50/MNIST_TRAIN_60000_corrupt_px_392_unrot_lmdb/shuffled/"),
                  'test_net_out': '"{}"'.format(lmdb_root + "lmdb_corrupted_50/MNIST_TEST_10000_corrupt_px_392_unrot_lmdb/shuffled/") 
                 }
                }
          }

for c in cases:
    print(c)
    set_up_dir(root + c + '/snapshots/')
    set_up_dir(root + c + '/log/')
    net_file =  root + c + '/net.prototxt'
    solver_file = root + c + '/solver.prototxt'
    
    replacement_dict_solver['network'] = str('"{}"'.format(net_file))
    replacement_dict_solver['snapshotprefix'] = str('"{}"'.format(root + c + '/snapshots/'))
    
    replacement_dict = cases[c]['replacement_dict']
    
    edit_prototxt(replacement_dict, cases[c]['template_net'], net_file)
    edit_solver(replacement_dict_solver, cases[c]['template_solver'], solver_file )

    cmd = caffe_root + ' train' + ' -solver ' + solver_file + ' -weights ' + init_weights + ' 2>&1 | tee -a ' + root + c + '/log/' + 'log.log'
    with open(root + c + "/cmd.txt", "w") as f:
        f.write(cmd)
    print(cmd)