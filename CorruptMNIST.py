import caffe
import lmdb
import numpy as np
import cv2
import io
import os
import numpy as np

class MNISTcorrupter(object):
    
    def __init__(self, src_lmdb_dir, dst_lmdb_dir, N_im, batch_size, set_str, dir_str, angles, N_pix_corrupt):
        self.src_lmdb_dir = src_lmdb_dir
        self.dst_lmdb_dir = dst_lmdb_dir
        self.set_str = set_str
        self.dir_str = dir_str
        self.NUM_IDX_DIGITS = 10
        self.IDX_FMT = '{:0>%d' % self.NUM_IDX_DIGITS + 'd}'  
        self.MAP_SZ = int(1e12)
        self.N_im = N_im  #Size of images used from MNIST
        self.batch_size = batch_size
        self.angles = angles
        self.N_pix_corrupt = N_pix_corrupt
        self.n_angles = len(angles)

    @staticmethod    
    def rotate_image(im, angle):
        rows,cols = im.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        return cv2.warpAffine(im,M,(cols,rows))

        
    @staticmethod
    def corrupt_image(im, N_pix_corrupt):
        rows,cols = im.shape
        randint = np.random.choice(int(rows*cols), N_pix_corrupt, replace=False)
        im2 = np.copy(im.flatten())
        im2[randint] = 0.
        return randint, im2.reshape((rows,cols))

    def shuffle_samples_lmdb(self,path_lmdb, path_dst, keys):
        """
        Copy select samples from an lmdb into another.
        Can be used for sampling from an lmdb into another and generating a random shuffle
        of lmdb content.

        Parameters:
        path_lmdb -- source lmdb
        path_dst -- destination lmdb
        keys -- list of keys or indices to sample from source lmdb
        """

        db = lmdb.open(path_dst, map_size=self.MAP_SZ)
        key_dst = 0
        with db.begin(write=True) as txn_dst:
            with lmdb.open(path_lmdb, readonly=True).begin() as txn_src: 
                for key_src in keys:
                    ########################
                    if not isinstance(key_src, basestring):
                        key_src = self.IDX_FMT.format(key_src)
                    txn_dst.put(self.IDX_FMT.format(key_dst), txn_src.get(key_src))
                    key_dst += 1


                    ########################

                #if not isinstance(key_src, basestring):
                    #    key_src = IDX_FMT.format(key_src)
                    #    
                    #datum_from = caffe.proto.caffe_pb2.Datum()#

                #    datum_from.ParseFromString(txn_src.get(key_src))

                 #   datum_to = caffe.proto.caffe_pb2.Datum()
                   # datum_to.label = datum_from.label
                  #  datum_to.data = datum_from.data

                    #txn_dst.put(IDX_FMT.format(key_dst),  datum_to.SerializeToString())
                    #key_dst += 1
                    # if key_dst ==5:
                    # break
        db.close()

   
    def create_shuffled_ind(self, lmdb_dir, txt_save_dir):
        db = lmdb.open(lmdb_dir)
        N_im = db.stat()['entries']
        ind = range(N_im)
        print('Indices created', N_im)
        np.random.shuffle(ind)
        keys_shuffled = [self.IDX_FMT.format(i) for i in ind]

        import json
        if not os.path.exists(txt_save_dir):
            os.makedirs(txt_save_dir)

        f = open(txt_save_dir + '/indices_list.txt', 'w')
        json.dump(keys_shuffled, f)
        f.close()
        return ind

    def load_N_MNIST_images(self, N, path):
        #path = '/mnt/antares_raid/home/oliver/adhara/src/caffe/examples/mnist/mnist_train_lmdb'
        lmdb_env = lmdb.open(path)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum_db = caffe.proto.caffe_pb2.Datum()
        I = []
        Label = []
        im_count = 0
        for key, value in lmdb_cursor:
                datum_db.ParseFromString(value)
                label = datum_db.label
                data = caffe.io.datum_to_array(datum_db)
                im = data.astype(np.uint8)
                im = np.transpose(im, (2, 1, 0)) # original (dim, col, row)
                I.append(im.squeeze())
                Label.append(datum_db.label)
                im_count = im_count + 1

                if im_count >= N:
                    break
        return I, Label
    
    
    def run(self):
        set_st = self.set_str + str(self.N_im) + '_corrupt_px_%i_'%self.N_pix_corrupt 

        #Create angles

        #Load MNIST Database
        IMG, LABEL = self.load_N_MNIST_images(self.N_im, self.src_lmdb_dir + 'mnist' + self.dir_str + 'lmdb')
       # N_im = db.stat()['entries']


        #N_im = lmdb_env.stat()['entries']
        N = np.dot(np.shape(self.angles),self.N_im)
        N = N[0]


        map_size = N*28*28*100

        rot_angs = np.zeros(N)
        im_count = 0
        #Looping over batches
        for idx in range(int(np.ceil(N/self.batch_size))):
            count = 0
            env_randint = lmdb.open(self.dst_lmdb_dir + 'MNIST' + set_st + 'randint_lmdb/', map_size=map_size)
            env_rot = lmdb.open(self.dst_lmdb_dir + 'MNIST' + set_st + 'rot_lmdb/', map_size=map_size)
            env_rot_ang = lmdb.open(self.dst_lmdb_dir + 'MNIST' + set_st + 'rot_ang_lmdb/',map_size=int(1e12))
            env_unrot_ang = lmdb.open(self.dst_lmdb_dir + 'MNIST' + set_st + 'unrot_lmdb/',map_size=map_size)
            with env_rot.begin(write=True) as txn:    
                with env_randint.begin(write=True) as txn_randint:   
                    with env_unrot_ang.begin(write=True) as txn_unrot_ang:
                        with env_rot_ang.begin(write=True) as txn_rot_ang:
                            # Looping over images from MNIST
                            for  in_, lab_ in zip(IMG[(int(self.batch_size*idx)):(int(self.batch_size*(idx+1)))], LABEL[int((self.batch_size*idx)):(int(self.batch_size*(idx+1)))]):
                                im = in_
                                label = lab_
                                for angle in self.angles:
                                    #Prepare data
                                    X = self.rotate_image(im.squeeze(), angle)
                                    randint, X = self.corrupt_image(X, self.N_pix_corrupt)
                                    X = np.reshape(X,[1,28,28])

                                    datum = caffe.io.array_to_datum(X.astype(float), label)
                                   #  if in_idx ==1:
                                   #     print(X.astype(float).shape)
                                    #datum.channels = X.shape[0]
                                    #datum.height = X.shape[1]
                                    #datum.width = X.shape[2]
                                    #datum.data = X.tostring()  # or .tostring() if numpy < 1.9

                                    #datum.label = label
                                    rot_angs[count] = str(angle) # "{0}".format(angle)
                                    str_id = self.IDX_FMT.format(int(self.batch_size*self.n_angles*idx) + count)
                                    #if count%100==0:
                                    #    print(str_id)
                                    #print(datum)

                                    # The encode is only essential in Python 3
                                    txn.put(str_id, datum.SerializeToString())
                                    txn_rot_ang.put(str_id, str(angle) )

                                    #Storing unroted images
                                    datum_unrot = caffe.io.array_to_datum(np.reshape(im.squeeze(),[1,28,28]).astype(float), label)

                                    txn_unrot_ang.put(str_id, datum_unrot.SerializeToString()) 
                                    # Storing corrupted randints
                                    txn_randint.put(str_id, str(randint)) 
                                    count = count + 1
                                    if int(self.batch_size*self.n_angles*idx + count)%10000==0:
                                        print('Image Nr. %i created from %s'%(int(self.batch_size*self.n_angles*idx + count),self.dir_str))
                        env_rot_ang.close()
                    env_unrot_ang.close()
                env_randint.close()
            env_rot.close()

        src_dir = self.dst_lmdb_dir + 'MNIST' + set_st + 'rot_lmdb'
        dst_dir = self.dst_lmdb_dir + 'MNIST' + set_st + 'rot_lmdb/shuffled'

        ind = self.create_shuffled_ind(src_dir, dst_dir)

        self.shuffle_samples_lmdb(src_dir, dst_dir, ind)

        src_dir = self.dst_lmdb_dir + 'MNIST' + set_st + 'rot_ang_lmdb'
        dst_dir = self.dst_lmdb_dir + 'MNIST' + set_st + 'rot_ang_lmdb/shuffled'
        self.shuffle_samples_lmdb(src_dir, dst_dir, ind)

        src_dir = self.dst_lmdb_dir + 'MNIST' + set_st + 'unrot_lmdb'
        dst_dir = self.dst_lmdb_dir + 'MNIST' + set_st + 'unrot_lmdb/shuffled'
        self.shuffle_samples_lmdb(src_dir, dst_dir, ind)

        
if __name__ == '__main__':
    transformlmdb = MNISTtransformer()
        
    
    
