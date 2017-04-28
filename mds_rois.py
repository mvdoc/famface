import pylab as pl
import numpy as np

import scipy.stats as stats
import scipy.spatial.distance as dist

from mvpa2.datasets import dataset_wizard, vstack
from mvpa2.testing.datasets import datasets as testing_datasets
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.misc.fx import get_random_rotation
from mvpa2.mappers.zscore import zscore

from mvpa2.mappers.procrustean import ProcrusteanMapper

from itertools import product as iproduct

from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt

# MDS implementation (we use metric one) from sklearn -- iterative
# optimization, high variability across "runs"

# Classical MDS interfaced from R
import rpy2.robjects as ro
from rpy2.robjects.numpy2ri import numpy2ri
ro.conversion.py2ri = numpy2ri
def rcmdscale(a, k=2, eig=False):
    res = ro.r['cmdscale'](a, eig=eig, k=k)
    if isinstance(res, ro.vectors.ListVector):
       assert(len(res) == 5)
       return [np.array(x) for x in res]
    return np.array(res)

mdsf = rcmdscale
def mds_withprocrust(a, t, **kwargs):
    # data should already be in the needed scale -- we just care about
    # rotation, shift, reflection
    pm = ProcrusteanMapper(
     reflection=True, scaling=False, reduction=False, oblique=False)
    
    a_ = mdsf(a, **kwargs)
    ds = dataset_wizard(a_, targets=t)
    pm.train(ds)
    return pm.forward(a_)

def xboost(l, n=100):
    for i in xrange(n):
        # Get l elements with "replacement"
        yield np.random.randint(0, l, l)


# We will create a bunch of subjects based on some canonical dataset
# but randomly rotating the patterns (so we are not in the same
# 'space'), adding some noise, and subselecting features
def get_fake_data(nsubjects=20, noise_level=0.2, nbogus_classes=0):
    orig_ds = mean_group_sample(['targets'])(testing_datasets['uni3large'])
    # and creating an additional target which is a composition of the other two, so
    # it should be closer to them than to the left out L2
    classes_data = [orig_ds.samples,
                   orig_ds[0].samples + orig_ds[1].samples,
                   orig_ds[1].samples + 4*orig_ds[2].samples]
    classes_targets = list(orig_ds.T) + ['L0+1', 'L1+4*2']
    if nbogus_classes:
       classes_data.append(np.zeros((nbogus_classes, classes_data[0].shape[1]), dtype=float))
       classes_targets += ['B%02d' % i for i in xrange(nbogus_classes)]
    proto_ds = dataset_wizard(np.vstack(classes_data), targets=classes_targets)
    ntargets = len(proto_ds.UT)
    dss = []
    for i in xrange(nsubjects):
        R = get_random_rotation(proto_ds.nfeatures)
        ds = dataset_wizard(np.dot(proto_ds.samples, R), targets=proto_ds.T)
        #ds = dataset_wizard(proto_ds.samples, targets=proto_ds.T)
        ds.sa['subjects'] = [i]
        # And select a varying number of features
        ds = ds[:, :np.random.randint(10, ds.nfeatures)]
        # Add some noise
        ds.samples += np.random.normal(size=ds.shape) * noise_level
        dss.append(ds)
    return dss

def get_fake_rois_data(nsubjects=20, nroi=5, **kwargs):
    dss = []
    for isubject in xrange(nsubjects):
        dss.append(get_fake_data(nroi, **kwargs))
    return dss

# get dissimilarities across rois
def get_dissimilarities(dss_subjects_rois, roi_labels=None):
    dss = []
    for dss_rois in dss_subjects_rois:
        dissimilarities_rois = np.array([dist.pdist(ds, 'correlation') for ds in dss_rois])
        # and those would compose our 'dss'
    if roi_labels is None:
       roi_labels = ['ROI%d' % i for i in xrange(len(dissimilarities_rois))]
       dss.append(dataset_wizard(dissimilarities_rois, targets=roi_labels))
    return dss


def bootstrap_mds(dss, k=2, nbootstraps=100, procrust=True, title="", 
        colors=None, compute_dissims=True, metric='correlation'):
    if compute_dissims:
        # compute dissimilarities among provided datasets
        dissimilarities = np.array([dist.squareform(dist.pdist(ds, metric)) for ds in dss])
        ntargets = len(dss[0].UT)
        nsubjects = len(dss)
        assert(dissimilarities.shape == (nsubjects, ntargets, ntargets))
    else:
        dissimilarities = np.array([ds.samples for ds in dss])
        assert(dissimilarities.shape[0] == len(dss))
    # full training/fit
    orig_fit = mdsf(np.mean(dissimilarities, axis=0), k=k)
    # full bootstrap fits
    stitle = title + " MDS"
    if procrust:
       mds_ = lambda x: mds_withprocrust(x, orig_fit, k=k)
       stitle += " + procrustean"
    else:
       mds_ = mdsf
    boost_full_fits = np.array([mds_(np.mean(dissimilarities[sel], axis=0))
                                for sel in xboost(len(dss), nbootstraps)])
    plot_bootstraps(orig_fit, boost_full_fits, dss[0].T, colors=colors);
    pl.title(stitle + " all classes");
    #if len(orig_fit) > 4:
    #   plot_bootstraps(orig_fit[:4], boost_full_fits[:, :4], dss[0].T[:4]); pl.title(stitle + " only meaningful");


def dendrogram_dss(dss, metric='correlation'):
    import mvpa2.base.dataset
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    # probably better mean of dissimilarities than dissimilarity of the mean
    # dss_mean = mean_group_sample(['targets'])(mvpa2.base.dataset.vstack(dss))
    
    # verify that they are all in the same order
    dss_rois = dss[0].T
    for ds in dss: assert(np.all(ds.T == dss_rois))
    dss_mean_diss = np.mean([dist.pdist(ds, metric) for ds in dss], axis=0)
    dss_mean_diss_ = squareform(dss_mean_diss)
    # sort them according to caudal to rostral
    if 'centers' in dss[0].sa:
        order = np.argsort(dss[0].sa.centers[:, 1:], axis=0)[:, 0]
        dss_mean_diss_ = dss_mean_diss_[order, :][:, order]
        dss_rois = dss_rois[order]
    pl.figure(figsize=(12,7))
    pl.subplot(1,2,1)
    res = pl.imshow(dss_mean_diss_, interpolation='nearest')
    pl.xticks(np.arange(len(dss_rois)), dss_rois, rotation=90);
    pl.yticks(np.arange(len(dss_rois)), dss_rois);
    pl.subplot(1,2,2)
    dendrogram(linkage(dss_mean_diss_, method='complete'), labels=dss_rois, orientation='right');


# borrowed from statsmodels
from statsmodels.graphics.plot_grids import _make_ellipse

def plot_bootstraps(origin, bootstraps, labels, ellipse_level=0.95, scatter_alpha=0.5, colors=None):
    fig = pl.figure(figsize=(14,10))
    ax = pl.axes(aspect=1)
    #ax = pl.axes([-1, -1, 1, 1])
    default_colors = "bgrcmyk" # TODO
    for i, l in enumerate(labels):
        # random color -- do better than random but so they stay distinct and
    # not limited by predefined set of colors
    # c = np.random.rand(3,1)
        c = (colors or default_colors)[i % len(default_colors)]
    # scatter plot all the bootstrap samples
        bs = bootstraps[:, i]
        if scatter_alpha:
            ax.scatter(bs[:, 0], bs[:, 1], c=c[0], alpha=scatter_alpha)
        # plot original 'center'
        x, y = origin[i][:2]
        ax.plot(x, y, 'o', label=l, markeredgewidth=2, markerfacecolor=c[0])
        ax.text(x, y, l, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8),
            color=c[0],
            horizontalalignment='center', verticalalignment='center')
        # plot ellipse
        if ellipse_level:
            # estimate mean/cov for bootstraps
            _make_ellipse(bs.mean(0)[:2], np.cov(bs, rowvar=0)[:2, :2], ax, level=ellipse_level, color=c)
    # and now plot all the sample points we got
    #break
    # pl.legend()
    pl.autoscale(tight=False)

    #ax.set_xlim([-1, 1])
    #ax.set_ylim([-1, 1])
    #ax.set_autoscaley_on(False)
    #ax.set_autoscalex_on(False)


def plot_summary_dissimilarities(dss):
    """Helper to look at actual dissimilarities and their correlation across subjects

    Was used to troubleshoot "unreleasticly" clean results for identities in famfaces
    whenever due to a bug data was loaded from the same subject
    """
    dissimilarities = np.array([dist.pdist(ds, 'correlation') for ds in dss]);
    c = np.corrcoef(dissimilarities, rowvar=1)
    pl.figure(figsize=(14,6));
    pl.subplot(1,3,1); pl.imshow(dissimilarities, interpolation='nearest'); pl.title('dissimilarities'); pl.colorbar();
    pl.subplot(1,3,2); pl.imshow(c, interpolation='nearest'); pl.title('corr(dissimilarities)'); pl.colorbar();
    pl.subplot(1,3,3); pl.hist(c[np.triu_indices(len(c), 1)], bins=21); pl.title('hist(corr(dissimilarities))');
    pl.figure(figsize=(14,6));
    c_pairs = np.corrcoef(dissimilarities, rowvar=0)
    pl.subplot(1,3,2); pl.imshow(c_pairs, interpolation='nearest'); pl.title('corr(dissimilarities.T)'); pl.colorbar();
    pl.subplot(1,3,3); pl.hist(c_pairs[np.triu_indices(len(c_pairs), 1)]); pl.title('hist(corr(dissimilarities.T))')


## load dataset and return dissimilarities in each roi
def get_dissim_roi(subnr):
    ds = h5load(fns.betafn(subnr))
    ds = ds[:, mask_]
    ds = ds[ds.sa.condition != 'self']
    zscore(ds, chunks_attr='chunks')
    ds = mean_group_sample(['condition'])(ds)
    
    names = []
    dissims = []
    for roi, (center, ids) in rois.iteritems():
        names.append(roi)
        sample_roi = ds.samples[:, ids]
        dissim_roi = pdist(sample_roi, 'correlation')
        dissims.append(dissim_roi)
    dss = dataset_wizard(dissims, targets=names)
    return dss


from mvpa2.generators.partition import OddEvenPartitioner
# from surf_anal_dsmsquared.py
def standardize(x, copy=True):
    if copy:
        x = np.asanyarray(x).copy();
    x -= x.mean(axis=0); x /= np.std(x, axis=0);
    return x

def corrcoefxy(x, y, already_standardized=False, fisher=False):
    """Corrcoef which would not waste memory on within each argument correlations

    Altogether thus is tiny bit faster than stock corrcoef, but also
    would need to double memory first for standardization
    """
    if not already_standardized:
        x = standardize(x)
        y = standardize(y)
    out = np.dot(x.T, y)/(len(x))
    if fisher:
        out = np.arctanh(out)
    return out


## load dataset and return dissimilarities in each roi
def get_dsm_roi_secondorder_xval2(ds, rois, zscore_ds=True, 
        part=OddEvenPartitioner(), cond_chunk='condition'):
    """ Obtain second-order dissimilarities between ROIs. This version
    cross-validates at the second level, thus the resulting dsms are 
    not symmetrical.

    Arguments
    --------
    ds: dataset
    rois: dict
        each item in the dictionary must be a tuple where the 0th element is
        the center of the roi, and the 1st element is a list of ids
    zscore_ds: bool
        is the dset already zscored?
    part: partitioner
    cond_chunk: str
        across which sample attribute to perform mean group sample

    Returns
    -------
    dataset containing second level dsm
    """
    #ds = h5load(fns.betafn(subnr))
    #ds = ds[:, mask_]
    #ds = ds[ds.sa.condition != 'self']
    if zscore_ds:
        zscore(ds, chunks_attr='chunks')
    
    # set up oddeven partition
    #part = OddEvenPartitioner()
    
    rdms = []
    mgs = mean_group_sample([cond_chunk])
    dissims_folds = []
    for ds_ in part.generate(ds):
        ds_1 = ds_[ds_.sa.partitions == 1]
        ds_2 = ds_[ds_.sa.partitions == 2]
        
        ds_1 = mgs(ds_1)
        ds_2 = mgs(ds_2)
        assert(ds_1.samples.shape == ds_2.samples.shape)
        
        # first generate first-order rdms for each fold
        names = []
        centers = []
        dissims_1 = []
        dissims_2 = []
        for roi, (center, ids) in rois.iteritems():
            names.append(roi)
            centers.append(center)
            
            sample1_roi = ds_1.samples[:, ids]
            sample2_roi = ds_2.samples[:, ids]
            
            dissim1_roi = pdist(sample1_roi, 'correlation')
            dissim2_roi = pdist(sample2_roi, 'correlation')
            
            dissims_1.append(dissim1_roi)
            dissims_2.append(dissim2_roi)
            
        dss1 = np.array(dissims_1)
        dss2 = np.array(dissims_2)
        
        # now compute second-order rdm correlating across folds
        dissim_2ndorder = 1. - corrcoefxy(dss1.T, dss2.T)
        dissim_2ndorder = dataset_wizard(dissim_2ndorder, 
                                         targets=names)
        dissim_2ndorder.sa['centers'] = centers
        # also add fa information about roi
        dissim_2ndorder.fa['roi'] = names
        dissims_folds.append(dissim_2ndorder)
    
    # average
    dissims = dissims_folds[0]
    for d in dissims_folds[1:]:
        dissims.samples += d.samples
    dissims.samples /= len(dissims_folds)
    return dissims


# In[307]:

## load dataset and return dissimilarities in each roi -- second option
def get_dsm_roi_xval1(ds, rois, zscore_ds=True, 
        part=OddEvenPartitioner(), cond_chunk='condition'):
    """ Obtain second-order dissimilarities between ROIs. This version
    cross-validates at the first level, thus the resulting dsms are 
    symmetrical.

    Arguments
    --------
    ds: dataset
    rois: dict
        each item in the dictionary must be a tuple where the 0th element is
        the center of the roi, and the 1st element is a list of ids
    zscore_ds: bool
        is the dset already zscored?
    part: partitioner
    cond_chunk: str
        across which sample attribute to perform mean group sample

    Returns
    -------
    dataset containing second level dsm
    """
    #ds = h5load(fns.betafn(subnr))
    #ds = ds[:, mask_]
    #ds = ds[ds.sa.condition != 'self']
    if zscore_ds:
        zscore(ds, chunks_attr='chunks')
    
    # set up oddeven partition
    #part = OddEvenPartitioner()
    
    rdms = []
    mgs = mean_group_sample([cond_chunk])
    dissims_folds = []
    for ds_ in part.generate(ds):
        ds_1 = ds_[ds_.sa.partitions == 1]
        ds_2 = ds_[ds_.sa.partitions == 2]
        
        ds_1 = mgs(ds_1)
        ds_2 = mgs(ds_2)
        assert(ds_1.samples.shape == ds_2.samples.shape)
        
        # first generate first-order rdms cross-validated across folds
        names = []
        centers = []
        dissims = []
        for roi, (center, ids) in rois.iteritems():
            names.append(roi)
            centers.append(center)
            
            sample1_roi = ds_1.samples[:, ids]
            sample2_roi = ds_2.samples[:, ids]
            
            dissim_roi = 1. - corrcoefxy(sample1_roi.T, sample2_roi.T)
            nsamples = ds_1.nsamples
            assert(dissim_roi.shape == (nsamples, nsamples))
            
            dissims.append(dissim_roi.flatten())  # now the RDM is not symmetrical anymore
        dissims_folds.append(np.array(dissims))
    
    # average across folds
    dissims_folds = np.array(dissims_folds).mean(axis=0)
    assert(dissims_folds.shape == (len(names), nsamples**2))

    # now compute second level (distances)
    distance_roi = dist.pdist(dissims_folds, metric='correlation')
    
    dissims_folds = dataset_wizard(dist.squareform(distance_roi),
                                   targets=names)
    dissims_folds.fa['roi'] = names
    dissims_folds.sa['centers'] = centers

    return dissims_folds


def get_dsm_roi_xval1_firstlev(ds, rois, zscore_ds=True, 
        part=OddEvenPartitioner(), cond_chunk='condition', fisher=False):
    """ Obtain second-order dissimilarities between ROIs. This version
    cross-validates at the first level and returns only the first level,
    without distances between ROIs

    Arguments
    --------
    ds: dataset
    rois: dict
        each item in the dictionary must be a tuple where the 0th element is
        the center of the roi, and the 1st element is a list of ids
    zscore_ds: bool
        is the dset already zscored?
    part: partitioner
    cond_chunk: str
        across which sample attribute to perform mean group sample
    fisher: bool
        whether to fisher-transform the correlations before averaging across folds

    Returns
    -------
    dataset containing first level dsm of shape (nrois, ncond**2)
    """
    #ds = h5load(fns.betafn(subnr))
    #ds = ds[:, mask_]
    #ds = ds[ds.sa.condition != 'self']
    
    # set up oddeven partition
    #part = OddEvenPartitioner()
    
    mgs = mean_group_sample([cond_chunk])
    dissims_folds = []
    folds = 1
    for ds_ in part.generate(ds):
        print("Running fold {0}".format(folds))
        ds_1 = ds_[ds_.sa.partitions == 1]
        ds_2 = ds_[ds_.sa.partitions == 2]
        
        ds_1 = mgs(ds_1)
        ds_2 = mgs(ds_2)
        if ds_1.nsamples >= 4 and zscore_ds:
            zscore(ds_1, chunks_attr='chunks')
            zscore(ds_2, chunks_attr='chunks')
        assert(ds_1.samples.shape == ds_2.samples.shape)
        
        # first generate first-order rdms cross-validated across folds
        names = []
        centers = []
        dissims = []
        for roi, (center, ids) in rois.iteritems():
            names.append(roi)
            centers.append(center)
            
            sample1_roi = ds_1.samples[:, ids]
            sample2_roi = ds_2.samples[:, ids]
            
            dissim_roi = corrcoefxy(sample1_roi.T, sample2_roi.T, fisher=fisher)
            nsamples = ds_1.nsamples
            assert(dissim_roi.shape == (nsamples, nsamples))
            
            dissims.append(dissim_roi.flatten())  # now the RDM is not symmetrical anymore
        dissims_folds.append(np.array(dissims))
        folds += 1
    
    # average across folds
    dissims_folds = np.array(dissims_folds).mean(axis=0)
    assert(dissims_folds.shape == (len(names), nsamples**2))
    
    if fisher:
        dissims_folds = np.tanh(dissims_folds)

    dissims_folds = dataset_wizard(dissims_folds,
                                   targets=names)
    dissims_folds.sa['centers'] = centers

    return dissims_folds


def get_minmax(array, how='maxabs'):
    import scipy
    if how == 'maxabs':
        absmax = np.abs(array).max()
        vlim = [-absmax, absmax]
    elif how == 'minmax':
        vlim = [array.min(), array.max()]
    elif how == 'minmaxsat':
        vlim = [scipy.stats.scoreatpercentile(array, 2),
                scipy.stats.scoreatpercentile(array, 98)]
    else:
        raise ValueError('Who knows about {0}?'.format(how))
    return vlim


def dendrogram_dss_mvdoc(dss, compute_distance=False, metric='correlation', 
        vlim='minmax'):
    """
    set compute_distance to True if dss contain 1st-level
    """
    import mvpa2.base.dataset
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import dendrogram, linkage
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # probably better mean of dissimilarities than dissimilarity of the mean
    # dss_mean = mean_group_sample(['targets'])(mvpa2.base.dataset.vstack(dss))
    
    # verify that they are all in the same order
    dss_rois = dss[0].T
    for ds in dss: assert(np.all(ds.T == dss_rois))
    if compute_distance:
        dss_mean_diss = np.mean([dist.pdist(ds, metric) for ds in dss], axis=0)
        dss_mean_diss_ = squareform(dss_mean_diss)
    else:
        dss_mean_diss_ = np.mean(np.dstack(dss), axis=-1)
    # sort them according to caudal to rostral
    if 'centers' in dss[0].sa:
        order = np.argsort(dss[0].sa.centers[:, 1:], axis=0)[:, 0]
        dss_mean_diss_ = dss_mean_diss_[order, :][:, order]
        dss_rois = dss_rois[order]
    fig = pl.figure(figsize=(12,7))
    ax = fig.add_subplot(1,2,1)
    if not isinstance(vlim, list):
        vlim_ = get_minmax(dss_mean_diss_.flatten(), how=vlim)
    else:
        vlim_ = vlim
    res = ax.imshow(dss_mean_diss_, interpolation='nearest', 
            vmin=vlim_[0], vmax=vlim_[1])
    pl.xticks(np.arange(len(dss_rois)), dss_rois, rotation=90);
    pl.yticks(np.arange(len(dss_rois)), dss_rois);
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    pl.colorbar(res, cax=cax)
    fig.add_subplot(1,2,2)
    dendrogram(linkage(dss_mean_diss_, method='complete'), labels=dss_rois, orientation='right');


from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_dsm(dsm, labels):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    vmax = np.round(dsm.max(), 2)
    res = ax.imshow(dsm, interpolation='nearest', vmin=0, vmax=vmax)
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.tick_params(length=0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = plt.colorbar(res, cax=cax, ticks=np.round(np.linspace(0, vmax, 8), 2))
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")
    return fig


def select_rois_pymvpa(ds, rois):
    return ds.select(sadict={'targets': rois}, fadict={'roi': rois})


from mpl_toolkits.mplot3d import Axes3D
def plot_mds_3d(coords, labels, sig_connection=None, dist_orig=None, view=(30, 30), 
        cmap=cm.summer, vlim='maxabs', plot_corr=True):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    if sig_connection is not None and dist_orig is not None:
        assert(sig_connection.ndim == 2)
        assert(dist_orig.ndim == 2)
        assert(sig_connection.shape == dist_orig.shape)
        if plot_corr:
            corr_orig_utr = 1. - dist.squareform(dist_orig)
        else:
            corr_orig_utr = dist.squareform(dist_orig)
        corr_orig = dist.squareform(corr_orig_utr)
        sig_connection_utr = dist.squareform(sig_connection).astype(bool)

        if isinstance(vlim, str):
            significant_upper_tr = corr_orig_utr[np.where(sig_connection_utr)]
            mind, maxd = get_minmax(significant_upper_tr, vlim)
        elif vlim is not None:
            mind, maxd = vlim
        print mind, maxd
        
        norm = mpl.colors.Normalize(vmin=mind, vmax=maxd)
        #cmap = cm.jet#cm.Blues#hot
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for label, (x, y, z), sig_conn, orig_strength in zip(labels, coords, sig_connection, corr_orig):
            # find out which we need to plot
            idx_sig = np.where(sig_conn)
            coords_sig = coords[idx_sig]
            strength_sig = orig_strength[idx_sig]
            #if len(coords_sig) == 0:
            #    print "Removing {0} because of no significant connections".format(label)
            #    continue
            ax.scatter(x, y, z, label=label, alpha=0)
            ax.text(x, y, z, label, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='center', 
                verticalalignment='center')
            for (xs, ys, zs), strength in zip(coords_sig, strength_sig):
                ax.plot([x, xs], [y, ys], [z, zs], '-', color=m.to_rgba(strength), linewidth=2, alpha=.5)
    else:
        for label, (x, y, z) in zip(labels, coords):
            ax.scatter(x, y, z, label=label, alpha=0)
            ax.text(x, y, z, label, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='center', 
                verticalalignment='center')
    ax.view_init(*view) 
    return fig


# In[39]:

def plot_mds_2d(coords, labels, sig_connection=None, dist_orig=None, 
        cmap=cm.summer, vlim='maxabs', plot_corr=True):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    if sig_connection is not None and dist_orig is not None:
        assert(sig_connection.ndim == 2)
        assert(dist_orig.ndim == 2)
        assert(sig_connection.shape == dist_orig.shape)
        if plot_corr:
            corr_orig_utr = 1. - dist.squareform(dist_orig)
        else:
            corr_orig_utr = dist.squareform(dist_orig)
        corr_orig = dist.squareform(corr_orig_utr)
        sig_connection_utr = dist.squareform(sig_connection).astype(bool)

        if isinstance(vlim, str):
            significant_upper_tr = corr_orig_utr[np.where(sig_connection_utr)]
            mind, maxd = get_minmax(significant_upper_tr, vlim)
        elif vlim is not None:
            mind, maxd = vlim
        print mind, maxd
        
        norm = mpl.colors.Normalize(vmin=mind, vmax=maxd)
        #cmap = cm.jet#cm.Blues#hot
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        for label, (x, y), sig_conn, orig_strength in zip(labels, coords, sig_connection, corr_orig):
            # find out which we need to plot
            idx_sig = np.where(sig_conn)
            coords_sig = coords[idx_sig]
            strength_sig = orig_strength[idx_sig]
            #if len(coords_sig) == 0:
            #    print "Removing {0} because of no significant connections".format(label)
            #    continue
            ax.scatter(x, y, label=label, alpha=0)
            ax.text(x, y, label, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='center', 
                verticalalignment='center')
            for (xs, ys), strength in zip(coords_sig, strength_sig):
                ax.plot([x, xs], [y, ys], '-', color=m.to_rgba(strength), linewidth=2, alpha=.5)
    else:
        for label, (x, y) in zip(labels, coords):
            ax.scatter(x, y, label=label, alpha=0)
            ax.text(x, y, label, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='center', 
                verticalalignment='top')
    return fig


def get_significant_labels(roi, labels_rois, sig_connection):
    assert(sig_connection.ndim == 2)
    roi_idx = np.where(roi == labels_rois)[0][0]
    return([labels_rois[i] for i in np.where(sig_connection[roi_idx, :])[0]])


def mk_colorbar(vlim, label='Decoding Accuracy', cmap=cm.hot):
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])

    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    #cmap = plt.cm.hot
    vlim = np.round(vlim, 2)
    norm = mpl.colors.Normalize(vmin=vlim[0], vmax=vlim[1])

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    
    # we don't need much information, so choose the ticks carefully
    v = np.round(np.linspace(vlim[0], vlim[1], 5), 2)
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal',
                                    ticks=v)
    cb1.set_label(label)
    cb1.solids.set_rasterized(True)

    return fig

