#!/usr/bin/env python
import argparse
from mvpa2.suite import *
import nibabel as nib
from os.path import join as pjoin
from os import makedirs


def main(infiles, outdir):
    # store data
    dss = []
    for infile in infiles:
        print('Loading {0}'.format(infile))
        dss.append(fmri_dataset(infile))

    count_mask = np.sum([ds.samples > 0 for ds in dss], axis=0)
    assert(count_mask.max() == 33)
    perc_mask = count_mask / len(dss)
    union_mask = count_mask > 0
    masks = {'count_mask': count_mask,
             'perc_mask': perc_mask,
             'union_mask': union_mask}

    try:
        makedirs(outdir)
    except OSError:
        pass

    for fn, mask in masks.iteritems():
        fnout = fn + '_{0}sbjs.nii.gz'.format(len(dss))
        print('Saving {0}'.format(fnout))
        map2nifti(dss[0], data=mask).to_filename(pjoin(outdir, fnout))


if __name__ == '__main__':
    import sys
    description = 'Make union mask by taking masks and saving them. Outputs' \
                  'hdf5 and nifti images of boolean mask, percentage mask, and ' \
                  'count mask'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('input_files', nargs='+', help='input masks in nifti file')
    parser.add_argument('output_dir',  help='Where to save the union masks')

    parsed = parser.parse_args(sys.argv[1:])
    main(parsed.input_files, parsed.output_dir)
