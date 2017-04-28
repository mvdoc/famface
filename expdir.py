from os.path import join as ospj

class expdir(object):
    """I'm lazy so this returns the common paths used in my analyses"""
    def __init__(self, basedir=None):
        """
        basedir: default is path on hydra
        """
        if basedir is None:
            self.basedir = '/data/famface/openfmri'
        else:
            self.basedir = basedir
        self.l1dir = ospj(self.basedir, 'results/l1ants_final/model001/task001')
        self.l2dir = ospj(self.basedir, 'results/l2ants_final/model001/task001')
            
    @property
    def maskfn(self):
        return ospj(self.l2dir, 'subjects_all/mask/union_mask_33sbjs_80p.nii.gz')
    
    def betafn(self, subnr):
        return ospj(self.l1dir, 'sub{0:03d}/betas/run_mc_noisecomp/runall.hdf5'.format(subnr))
    
    def mvpadir(self, subnr):
        return ospj(self.l1dir, 'sub{0:03d}/mvpa'.format(subnr))
