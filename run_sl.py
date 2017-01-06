#!/usr/bin/env python
import sys
import argparse
from mvpa2.suite import *
from mvpa2.generators.partition import NFoldPartitioner, FactorialPartitioner
from mvpa2.generators.permutation import AttributePermutator
import nibabel
from itertools import *
from os.path import join as pjoin


def shuffle_sa(ds, sa='condition', rand_seed=None):
    """Shuffle sa attribute within each chunk"""
    rng = np.random.RandomState(rand_seed)
    shuffled_sa = []
    for chunk in ds.UC:
        mask_chunk = ds.sa.chunks == chunk
        shuffled_sa.extend(rng.permutation(ds[mask_chunk].sa[sa]))
    ds.sa[sa+'_old'] = ds.sa[sa]
    ds.sa[sa] = shuffled_sa
    return ds


def test_shuffle_sa():
    ds = Dataset(range(8), sa={'condition': range(4)*2,
                               'chunks': np.repeat([0, 1], 4)})
    ds1 = ds.copy(True)
    ds2 = ds.copy(True)

    ds1 = shuffle_sa(ds1, rand_seed=1)
    ds2 = shuffle_sa(ds2, rand_seed=1)

    # assert it is deterministic
    assert(np.array_equal(ds1.sa.condition, ds2.sa.condition))
    # assert it is different
    assert(not np.array_equal(ds1.sa.condition, ds.sa.condition))
    # assert they are balanced
    for chunk in ds.UC:
        mask_chunk = ds.sa.chunks == chunk
        assert(np.array_equal(np.unique(ds[mask_chunk].sa.condition),
                              np.unique(ds1[mask_chunk].sa.condition)))

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

def main(infile, outdir, radius, mask, zscoring, classification, derivs=True,
         debugging=False, permute=None, decoder='svm', errors=False):
    # gime more
    if debugging:
        debug.active += ["SLC"]
    print('Loading {0}'.format(infile))
    ds = h5load(infile)
    # check we have derivatives too
    if derivs and 'derivs' not in ds.fa:
        raise ValueError('Dataset {0} does not contain derivatives'.format(infile))

    # let's try familiar vs unfamiliar
    if classification in ['familiar_vs_unfamiliar',
                          'familiar_vs_unfamiliar-id',
                          'familiar_vs_unfamiliar-id-chunks',
                          'identity-all',
                          'identity-familiar',
                          'identity-unfamiliar']:
        ds = ds[ds.sa.condition != 'self']
        # permute if needed 
        if permute:
            if classification != 'familiar_vs_unfamiliar-id':
                ds = shuffle_sa(ds, rand_seed=permute)
            else:
                # for familiar_vs_unfamiliar-id we need a fancier perm
                perm = get_unique_combs(8)[permute - 1]
                perm = flatten(perm)
                unique_conds = np.unique(ds.sa.condition)
                mapperm = dict()
                for i, p in enumerate(perm):
                    mapperm[unique_conds[i]] = unique_conds[p]
                for i in range(ds.nsamples):
                    this_cond = ds.sa.condition[i]
                    ds.sa.condition[i] = mapperm[this_cond]
                print("USING PERMUTATION {0}".format(mapperm))
        ds.sa['familiarity'] = ['familiar' if 'friend' in a
                                else 'control'
                                for a in ds.sa.condition]
    else:
        raise NotImplementedError('Classification not implemented')

    # if we are using a dataset with derivatives but we don't want to use them
    # as features, extract only the non-derivatives features
    sfx = ''
    if 'derivs' in ds.fa and not derivs:
        ds = ds[:, ds.fa.derivs == 0]
        sfx += '_betaderivs'


    # set up clf and cv
    if decoder == 'svm':
        clf = LinearCSVMC()
    elif decoder == 'gnb':
        clf = GNB()
    else:
        raise ValueError('I have no clue about this classifier {0}'.format(decoder))

    if classification == 'familiar_vs_unfamiliar':
        ds.sa['targets'] = ds.sa['familiarity']
        partitioner = NFoldPartitioner()
    elif classification == 'familiar_vs_unfamiliar-id':
        ds.sa['targets'] = ds.sa['familiarity']
        partitioner = FactorialPartitioner(NFoldPartitioner(attr='condition'),
                                           attr='targets')
        #if permute:
        #    rng = np.random.RandomState(permute)
        #    permutator = AttributePermutator(['familiarity'],
        #            limit=['partitions', 'chunks'],
        #            rng=rng)
        #    partitioner = ChainNode([partitioner, permutator], space='partitions')
    elif classification == 'familiar_vs_unfamiliar-id-chunks':
        ds.sa['targets'] = ds.sa['familiarity']
        # to do within chunks cross-validation across identities
        partitioner = ChainNode(
            [FactorialPartitioner(NFoldPartitioner(attr='condition'),
                                  attr='familiarity'),
             ExcludeTargetsCombinationsPartitioner(k=1,
                                                   targets_attr='chunks',
                                                   space='partitions')],
            space='partitions')
    elif classification == 'identity-all':
        ds.sa['targets'] = ds.sa['condition']
        partitioner = NFoldPartitioner()
    elif classification == 'identity-familiar':
        ds.sa['targets'] = ds.sa['condition']
        ds = ds.select(sadict={'condition': ['friend' + str(i) for i in range(1, 5)]})
        assert(ds.nsamples == 44)
        partitioner = NFoldPartitioner()
    elif classification == 'identity-unfamiliar':
        ds.sa['targets'] = ds.sa['condition']
        ds = ds.select(sadict={'condition': ['control' + str(i) for i in range(1, 5)]})
        assert(ds.nsamples == 44)
        partitioner = NFoldPartitioner()

    cv = CrossValidation(clf, partitioner)

    if mask:
        mask_ds = fmri_dataset(mask)
        if derivs:
            assert(np.all(mask_ds.fa.voxel_indices == ds.fa.voxel_indices[ds.fa.derivs == 0]))
        else:
            assert(np.all(mask_ds.fa.voxel_indices == ds.fa.voxel_indices))
        assert(len(mask_ds) == 1)
        mask_ = mask_ds.samples[0]  # extract mask as the first sample
        #assert(np.all(mask_ == mask_ds.samples.flatten()))
        if derivs:
            # need to make the mask bigger
            mask_ = np.tile(mask_, 2)
        ds = ds[:, mask_ > 0]
    if derivs:
        assert(np.all(ds.fa.voxel_indices[ds.fa.derivs == 0] == ds.fa.voxel_indices[ds.fa.derivs == 1]))
    #ds = remove_invariant_features(ds)
    # zscore within each chunk
    if zscoring:
        zscore(ds, chunks_attr='chunks', dtype='float32')

    # copy for efficiency
    ds_ = ds.copy(deep=False,
                 sa=['targets', 'chunks', 'familiarity', 'condition'],
                 fa=['voxel_indices', 'derivs'],
                 a=['mapper'])
    print(ds_)

    if derivs:
        sl = Searchlight(cv,
                         IndexQueryEngine(voxel_indices=Sphere(radius),
                                          derivs=Sphere(2)),
                         postproc=mean_sample(),
                         roi_ids=np.where(ds_.fa.derivs == 0)[0],
                         nproc=8)
    else:
        sl = sphere_searchlight(cv, radius=radius, space='voxel_indices',
                                #center_ids=range(0, 1000),
                                postproc=mean_sample(),
                                nproc=8
                                )

    # run it! -- oh, PyMVPA!
    sl_map = sl(ds_)
    # copy mapper
    sl_map.a = ds.a
    # remove unnecessary field to make file smaller
    del sl_map.a['add_regs'] 


    if not errors:
        sl_map.samples *= -1
        sl_map.samples += 1
    # reduce size
    sl_map.samples = sl_map.samples.astype('float32')

    # save
    fnout = 'sl'
    if mask:
        fnout += 'msk'
    if zscoring:
        fnout += 'z'
    fnout += str(radius) + 'vx'
    if derivs:
        fnout += '_featderivs'
        sfx = ''
    fnout += sfx
    fnout += '_' + decoder

    sl_out = pjoin(outdir, fnout, classification)
    try:
        os.makedirs(sl_out)
    except OSError:
        pass

    print('Saving in {0}'.format(sl_out))
    fnslmap = 'sl_map'
    if permute:
        fnslmap += '_perm{0:03d}'.format(permute)
    fnslmap += '.hdf5'
    h5save(pjoin(sl_out, fnslmap), sl_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--infile', '-i',
                        help='input file')
    parser.add_argument('--outdir', '-o',
                        help='output directory')
    parser.add_argument('--mask', '-m',
                        help='mask to be loaded', default=None)
    parser.add_argument('--zscore', '-z', action='store_true', default=False,
                        help='zscore the features')
    parser.add_argument('--derivs', action='store_true', default=False,
                        help='use derivatives as features')
    parser.add_argument('--errors', action='store_true', default=False,
                        help='return errors instead of accuracies')
    parser.add_argument('--slradius', '-r', type=int,
                        help='searchlight radius in voxels', default=3)
    parser.add_argument('--classification', '-c', type=str,
                        help='classification type',
                        choices=['familiar_vs_unfamiliar',
                                 'familiar_vs_unfamiliar-id',
                                 'familiar_vs_unfamiliar-id-chunks',
                                 'identity-all',
                                 'identity-familiar',
                                 'identity-unfamiliar'])
    parser.add_argument('--decoder', type=str,
                        help='classifier',
                        choices=['svm', 'gnb'],
default='svm')
    parser.add_argument('--debug', '-dbg', action='store_true', default=False,
                        help='whether to activate debug mode')
    parser.add_argument('--permute', default=None, type=int,
                        help='if set to an integer, it will randomly shuffle'
                             'the labels to generate a null distribution')
    parsed = parser.parse_args(sys.argv[1:])
    main(parsed.infile, parsed.outdir, parsed.slradius, parsed.mask, parsed.zscore,
         parsed.classification, derivs=parsed.derivs, debugging=parsed.debug,
         permute=parsed.permute, decoder=parsed.decoder, errors=parsed.errors)




