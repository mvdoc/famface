import argparse
from mvpa2.suite import *
import numpy as np
import sys
from os.path import join as pjoin
from os.path import dirname, basename
from os import makedirs
import glob


def main(infiles, outfile, derivs):
    sa_selection = [a + str(i) for a in ['control', 'friend']
                               for i in range(1, 5)] + ['self']

    dss = []
    # in case we use main from import
    if isinstance(infiles, str):
        infiles = [infiles]
    if len(infiles) == 1:  # assume globbing
        infiles = glob.glob(infiles[0])
    for i, fn in enumerate(sorted(infiles)):
        print('Loading {0} for stacking as chunk {1}'.format(fn, i))
        ds = h5load(fn)
        ds = ds.select(dict(condition=sa_selection))
        ds.sa['chunks'] = np.repeat(i, ds.nsamples)
        dss.append(ds)

    if derivs:
        # extract dataset
        sa_selection_extra = ['glm_label_' + a + '_derivative'
                for a in sa_selection]
        dss_extraregs = []
        for ds in dss:
            ds_ = ds.a.add_regs
            ds_ = ds_.select(dict(regressor_names=sa_selection_extra))
            ds_.sa['condition'] = [a.split('_')[2] 
                    for a in ds_.sa.regressor_names]
            # check the order is the same
            assert(np.array_equal(ds.sa.condition, ds_.sa.condition))
            dss_extraregs.append(ds_)
        dss_extraregs = vstack(dss_extraregs, a=0)
        dss_extraregs.fa['derivs'] = np.repeat(1, dss_extraregs.nfeatures)

    dss = vstack(dss, a=0)
    dss.fa['derivs'] = np.repeat(0, dss.nfeatures)
    if derivs:
        dss = hstack((dss, dss_extraregs), a=0)

    assert(dss.nsamples == 99)
    assert(np.array_equal(dss.UC, range(11)))

    parent = dirname(outfile)
    try:
        makedirs(parent)
    except OSError:
        pass

    print('Saving {0}'.format(outfile))
    h5save(outfile, dss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stack datasets')

    parser.add_argument('--infiles', '-i',
                        help='input files to stack', nargs='+')
    parser.add_argument('--outfile', '-o',
                        help='output file to save. if parent directory is inexistent, it will be created')
    parser.add_argument('--derivs',
            help='whether to hstack derivatives', action='store_true',
            default=False)
    parsed = parser.parse_args(sys.argv[1:])

    main(parsed.infiles, parsed.outfile, parsed.derivs)
