#!/usr/bin/env python
# coding: utf-8

import argparse
import sys
import expdir
from os.path import join as pjoin
import os
from mds_rois import get_dsm_roi_xval1_firstlev
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.base.hdf5 import h5save, h5load
from mvpa2.generators.partition import NFoldPartitioner, OddEvenPartitioner
from mvpa2.mappers.zscore import zscore

default_mask = '/data/famface/openfmri/results/l2ants_final/model001/task001/subjects_all/mask/union_mask_33sbjs_80p_MNI_cerebrum.nii.gz'
fns = expdir.expdir()


# In[5]:

def load_ds(subnr, mask=None, zscore_ds=True):
    ds = h5load(fns.betafn(subnr))
    if mask is not None:
        ds = ds[:, mask]
    ds = ds[ds.sa.condition != 'self']
    #if zscore_ds:
    #    zscore(ds, chunks_attr='chunks')
    # add familiarity
    ds.sa['familiarity'] = ['familiar' if l.startswith('f') else 'control' 
                            for l in ds.sa.condition]
    return ds


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rois', type=str,
            help='file containing rois',
            required=True)
    parser.add_argument('--subject', type=int,
            help='0-based subject number',
            required=True)
    parser.add_argument('--mask', type=str,
            help='mask to use -- probably not needed but keeping it',
            default=default_mask)
    parser.add_argument('--condition', type=str,
            help='what chunk to use', default='condition')
    parser.add_argument('--output', type=str,
            help='output directory template -- add sub{subnr:03d} and it will be formatted', required=True)

    return parser.parse_args()

def main():
    parsed = parse_args()
    roi_fn = parsed.rois
    maskfn = parsed.mask
    cond_chunk = parsed.condition
    subnr = parsed.subject + 1
    dirout = parsed.output

    if 'sub{subnr:03d}' not in dirout:
        raise ValueError('Wrong template for output directory. Please read help')

    # load mask
    mask = fmri_dataset(maskfn)
    mask_ = mask.samples[0] > 0

    print("Loading ROIs in {0}".format(roi_fn))
    rois = h5load(roi_fn)

    ds = load_ds(subnr, mask=mask_)

    part = NFoldPartitioner(5)
    zscore_ds = True
    if cond_chunk == 'familiarity':
        zscore_ds = False
    dsm = get_dsm_roi_xval1_firstlev(ds, rois, part=part,
                                     zscore_ds=zscore_ds, cond_chunk=cond_chunk,
                                     fisher=True)

    dirout = dirout.format(subnr=subnr)
    try:
        os.makedirs(dirout)
    except OSError:
        print('Directory exists, passing')

    print('Saving in {0}'.format(dirout))
    fnout = pjoin(dirout, 'dsm_{0}.hdf5'.format(cond_chunk))
    h5save(fnout, dsm)

if __name__ == '__main__':
    main()

