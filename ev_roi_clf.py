#!/usr/bin/env python
import sys
from os.path import join as pjoin
from os import makedirs
from collections import defaultdict
from itertools import chain
import pandas as pd

subnr = int(sys.argv[1])

paths = ['/home/contematto/github/PyMVPA/',
        '/data/famface/openfmri/scripts']

for p in paths:
    if p not in sys.path:
        sys.path.insert(1, p)

from expdir import expdir
fns = expdir()

from mvpa2.suite import *

# load union mask
#union_mask_fn = fns.maskfn
# use new union mask
union_mask_fn = '/data/famface/openfmri/results/l2ants_final/model001/task001/subjects_all/mask/union_mask_33sbjs_80p_MNI_cerebrum.nii.gz'
union_mask = fmri_dataset(union_mask_fn)
# load v1v_mask
#v1v_mask = fmri_dataset('/data/famface/openfmri/ProbAtlas_v4/v1v_mh.nii.gz')


def ev_mask_fn(which):
    mask_fn = '/data/famface/openfmri/ProbAtlas_v4/extracted_masks/{0}_mh_2mm.nii.gz'.format(which)
    return mask_fn


def load_mask(maskfn):
    mask = fmri_dataset(maskfn)
    mask_ = (union_mask.samples + mask.samples) == 2.
    return mask_.flatten()


# Setup permutation routines
def get_unique_combs(nsub):
    # assumes there are two superordinate categories, 
    # and that nsub is the total number of subordinate categories,
    # and that nsub is even
    combs = list(combinations(range(nsub), nsub/2))
    unique_combs = set()
    for c1, c2 in zip(combs, combs[::-1]):
        unique_combs.add(tuple(sorted((c1, c2))))
    return sorted(unique_combs)

def flatten(listOfLists):
    "Flatten one level of nesting"
    return list(chain.from_iterable(listOfLists))


def permute_conditions(ds, permute=0):
    """Permute the conditions in ds maintaining the hierarchical structure
    of the problem. Permute is the permutation number with max permute = 34.
    Permute = 0 corresponds to the identity permutation
    
    If it finds condition_orig in ds.sa, uses that as original condition 
    to permute"""
    perm = get_unique_combs(8)[permute]
    perm = flatten(perm)
    unique_conds = np.unique(ds.sa.condition)
    mapperm = dict()
    if 'condition_orig' in ds.sa:
        ds.sa['condition'] = ds.sa.condition_orig.copy()
    else:
        ds.sa['condition_orig'] = ds.sa.condition.copy()
    for i, p in enumerate(perm):
        mapperm[unique_conds[i]] = unique_conds[p]
    for i in range(ds.nsamples):
        this_cond = ds.sa.condition[i]
        ds.sa.condition[i] = mapperm[this_cond]
    print("USING PERMUTATION {0}".format(mapperm))


rois = ['V1v', 'V1d', 
        'V2v', 'V2d', 
        'V3v', 'V3d',
        'V1', 
        'V2', 
        'V3',
        'V1+V2',
        'V1+V2+V3',
       ]


mask_roi = {roi: load_mask(ev_mask_fn(roi)) for roi in rois}

for roi in rois:
    print('ROI {0}: {1:6d} voxels'.format(roi, mask_roi[roi].sum()))

def prepare_dataset(subnr, mask=None):
    ds = h5load(fns.betafn(subnr))
    if mask is not None:
        ds = ds[:, mask]
    ds = ds[ds.sa.condition != 'self', :]
    zscore(ds, chunks_attr='chunks', dtype='float32')
    
    return ds


# setup classifier
clf = LinearCSVMC()
# feature selection helpers
fselector = FractionTailSelector(.1,  # select top 10% features
                                 tail='upper',
                                 mode='select', sort=False)
sbfs = SensitivityBasedFeatureSelection(OneWayAnova(), fselector,
                                        enable_ca=['sensitivities'])
# create classifier with automatic feature selection
fsclf = FeatureSelectionClassifier(clf, sbfs)

partitioner = FactorialPartitioner(NFoldPartitioner(attr='condition'),
                                           attr='targets')

# setup cross-validation scheme
cv_anova = CrossValidation(fsclf, partitioner,
                     errorfx=lambda p, t: np.mean(p == t),
                     #enable_ca=['stats'],
                     #postproc=mean_sample()
                     )


# setup output dataframe
df_acc_roi = pd.DataFrame({'subid': [], 
                           'roi': [], 
                           'fold':[], 
                           'acc': [],
                           'permutation': []})


print("Running for subject {0}".format(subnr))
ds = prepare_dataset(subnr)
for iperm in range(35):  # iperm = 0 is the identity permutation
    permute_conditions(ds, permute=iperm)
    # set up targets
    ds.sa['familiarity'] = ['familiar' if 'friend' in a
                            else 'control'
                            for a in ds.sa.condition]
    ds.sa['targets'] = ds.sa['familiarity']
    for roi in rois:
        ds_ = ds[:, mask_roi[roi]]
        print('    Running on ROI {0}'.format(roi))
        acc_roi = cv_anova(ds_).samples.flatten()

        nfolds = len(acc_roi)
        tmp_df = pd.DataFrame(
            {'subid': ['sub{0:03d}'.format(subnr)] * nfolds,
             'roi': [roi] * nfolds,
             'fold': range(1, nfolds+1),
             'acc': acc_roi,
             'permutation': [iperm] * nfolds})
        df_acc_roi = pd.concat((df_acc_roi, tmp_df))


output_dir = '/data/famface/openfmri/results/ev-roi-class_newmask'
try:
    makedirs(output_dir)
except OSError:
    pass

df_acc_roi.to_csv(pjoin(output_dir, 'sub{0:03d}-evroi_acc+perm.csv'.format(subnr)), index=False)
