#!/usr/bin/env python
p = ['~/github/PyMVPA'
    ]
import argparse
from joblib import Parallel, delayed
import sys
for pp in p:
    if pp not in sys.path:
        sys.path.insert(1, pp)

from mvpa2.base.hdf5 import h5load
from mvpa2.datasets.cosmo import map2cosmo
from mvpa2.datasets.mri import fmri_dataset

# need to inject voxel_indices
mask = fmri_dataset('/data/famface/openfmri/results/l2ants_final/model001/task001/subjects_all/mask/union_mask_33sbjs_80p.nii.gz')
mask = mask[:, mask.samples[0] > 0]

def load_save(fn):
    print('Running on {0}'.format(fn))
    ds = h5load(fn)
    assert(len(mask.fa.voxel_indices) == len(ds.fa.center_ids))
    ds.fa['voxel_indices'] = mask.fa.voxel_indices
    _ = map2cosmo(ds, fn.replace('hdf5', 'mat'))

def main(filenames, nproc=8):
    Parallel(n_jobs=nproc)(delayed(load_save)(fn) for fn in filenames)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', nargs='+')
    parser.add_argument('--nproc', '-p', type=int, default=8)

    parsed = parser.parse_args(sys.argv[1:])
    main(parsed.infiles, nproc=parsed.nproc)
