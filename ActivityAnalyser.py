from layerwise_activations import store_layerwise_activations
from utils import set_up_dir, ReportInterface
import lmdb
import scipy
import cPickle as pickle

class ActivityAnalyser(object):
    def __init__(self,net_prototxt, model, phase, keys, n, dst_fpath, rot_lmdb_path ):
        self.net_prototxt = net_prototxt
        self.model = model
        self.phase = phase
        self.keys = keys
        self.n = n
        self.dst_fpath = dst_fpath
        self.rot_lmdb_path = rot_lmdb_path
        
    def __call__(self):
        # Get activations, store if not available
        print('Activations')
        S = ReportInterface()
        try:
            CL = S.__load_dict_from_hdf5__(self.dst_fpath)
        except IOError:
            store_layerwise_activations(self.net_prototxt, self.model, self.phase, self.keys, self.n, self.dst_fpath + 'activity.hdf5')
            print('Stores activations to {}'.format(self.dst_fpath))
            CL = S.__load_dict_from_hdf5__(self.dst_fpath)
            print('Stored and loaded activity')
        # Get rotation angles
        self.Rot, self.angs = self.get_rot_angles()
                  
        # Get KL_anglewise
        print('KL_anglewise')
        try:
            KL = pickle.load(open(self.dst_path + 'KL_anglewise.p','wb'))
        except IOError:
            print('Create KL_anglewise')
            self.store_KL_anglewise()
            KL = pickle.load(open(self.dst_path + 'KL_anglewise.p','wb'))
            print('Created KL_anglewise')
        # Get KL mean
        print('KL_mean')
        KL_clean, KL_mean = self.compute_KL_mean(KL)
    
        # Get selectivity score
        print('Selectivity_score')
        s_score = self.compute_selectivity_score(KL_clean, KL_mean)
                                                 
        return s_score
            
        
           
    def get_rot_angles(self):
        lmdb_env = lmdb.open(self.rot_lmdb_path)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        R = []
        im_count = 0
        for key, value in lmdb_cursor:
                R.append(value)
        Rot = np.asarray([float(r) for r in R])
        angs = np.sort(list(set(Rot)))
        return Rot, angs
                  
    def store_KL_anglewise(self):
          KL = {l:[[] for n,_ in enumerate(CL[layer].T)] for l in self.keys}
          for l in self.keys:        
              for neuron_nr, v in enumerate(CL[layer].T):
                  for r in self.angs:
                      p, _ = np.histogram(v[self.Rot==r],bins = 100, density = True) # p = P(act/rot)
                      q, _ = np.histogram(v[self.Rot!=r],bins = 100, density = True) # q = P(act)
                      KL[l][neuron_nr].append(scipy.stats.entropy(p,q))
          pickle.dump(open(self.dst_path + 'KL_anglewise.p','wb'), KL)
                  
                             
    def compute_KL_mean (self, KL):
        KL_mean =  {l:[] for l in self.keys}            
        KL_clean = {l:[] for l in self.keys}
        KL_naninf =  {l:[] for l in self.keys}
        for l in KL.keys():
            for neuron_nr, k in enumerate(KL[l]):
                if np.isinf(k).any() or np.isnan(k).any():
                    KL_naninf[l].append(neuron_nr)
                    pass
                else:
                    KL_clean[l].append(k/np.sum(k))

        #KL_clean = [np.asarray(k)[~np.isnan(np.asarray(k)) & ~np.isinf(np.asarray(k)) ] for k in KL]
            KL_clean[l] = np.asarray(KL_clean[l])


            KL_mean[l] = np.mean(KL_clean[l], axis = 0)
                             
        return KL_clean, KL_mean
                             
                     
    def compute_selectivity_score(self, KL_clean, KL_mean):
        s_score = {l:[] for l in self.keys}
        for l in KL_clean.keys():
            for p in KL_clean[l]:
                s_score[l].append(scipy.stats.entropy(p,KL_mean[l]))
        return s_score

                  
if __name__ == '__main__':
    print('ActivityAnalyser')
        