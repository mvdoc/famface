#! /usr/bin/env python
from mvpa2.suite import *
from os.path import join as pjoin
import glob
import argparse

basedir = '/data/famface/openfmri'
dsdir = 'data'
model = 'model001'
task = 'task001'
resultsdir = 'results/l1ants_final'


def get_motion_param(subnr, runnr):
    """ Returns a (nframes, 2) numpy array containing the regressors for motion """
    artdir = pjoin(basedir, l1dir, 'model001', 'task001', 'sub{0:03d}'.format(subnr), 'qa/art')
    # load motion parameters first
    fnmotion = pjoin(artdir, 'run{0:02d}_norm.bold_dtype_mcf.txt'.format(runnr))
    with open(fnmotion, 'r') as f:
        motion_params = [float(line) for line in f.readlines()]
    motion_params = np.array(motion_params)
    # now load outliers
    motion_outliers = np.zeros(motion_params.shape)
    fnoutliers = pjoin(artdir, 'run{0:02d}_art.bold_dtype_mcf_outliers.txt'.format(runnr))
    with open(fnoutliers, 'r') as f:
        # remove 1 because this is 0-based
        outliers_index = [int(line)-1 for line in f.readlines()]
    motion_outliers[outliers_index] = 1

    if np.sum(motion_outliers) == 0:
        return motion_params.reshape((-1, 1))
    else:
        return np.vstack((motion_params, motion_outliers)).T


def get_events(onsetdir, conditions):
    events = []
    for fn in glob.glob(pjoin(onsetdir, '*.txt')):
        cond = conditions[os.path.basename(fn).split('.')[0]]
        with open(fn, 'r') as f:
            for line in f.readlines():
                onset, duration, _ = line.split()
                events.append({'onset': float(onset),
                               'duration': float(duration),
                               'condition': cond})
    return events


def run_hrf_estimate(boldfn, onsetdir, condition_fn, derivs, output_model,
                     extra_regressors, extra_regressors_names=None):
    # load conditions
    conditions = {}
    with open(condition_fn, 'r') as f:
        for line in f.readlines():
            condnr, condtype = line.split()[1:]
            conditions[condnr] = condtype

    ds = fmri_dataset(boldfn)
    events = get_events(onsetdir, conditions)

    if derivs:
        hrf_model = 'canonical with derivative'
    else:
        hrf_model = 'canonical'

    betas = fit_event_hrf_model(ds, events, time_attr='time_coords',
                                condition_attr='condition',
                                design_kwargs=dict(hrf_model=hrf_model,
                                                   drift_model='blank',
                                                   add_regs=extra_regressors,
                                                   add_reg_names=extra_regressors_names),
                                glmfit_kwargs=dict(model='ols'),
                                return_model=output_model)

    return betas


def main(subject_number, run_number, derivs, noise_estimates, output_model):
    # get filenames
    subj = 'sub{0:03d}'.format(subject_number)
    run = 'run{0:03d}'.format(run_number)

    boldfn = pjoin(basedir, resultsdir, model, task, subj, 'bold', run, 'bold_mni.nii.gz')
    onsetdir = pjoin(basedir, dsdir, subj, 'model', model, 'onsets', task + '_' + run)
    condition_fn = pjoin(basedir, dsdir, 'models', model, 'condition_key.txt')

    if noise_estimates:
        noise_estimate_fn = pjoin(basedir, resultsdir, model, task, subj,
                                  'qa/noisecomp', 'run{0:02d}_noise_components.txt'.format(run_number))
        extra_regressors = np.loadtxt(noise_estimate_fn)
        print('Got {0} extra regressors'.format(extra_regressors.shape[1]))
    else:
        extra_regressors = None

    betas = run_hrf_estimate(boldfn, onsetdir, condition_fn, derivs,
                             output_model, extra_regressors)

    betas_dir = 'betas/run'
    if noise_estimates:
         betas_dir += '_mc_noisecomp'
    if derivs:
        betas_dir += '_derivs'
    if output_model:
        betas_dir += '_model'

    outdir = pjoin(basedir, resultsdir, model, task, subj, betas_dir)
    try:
        os.makedirs(outdir)
    except OSError:
        print('Directory exists, catching error')

    fnout = run
    fnout += '.hdf5'
    print('Saving {0}'.format(fnout))
    h5save(pjoin(outdir, fnout), betas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit canonical GLM model with '
                                                 'extra regressors. By default it uses'
                                                 'motion estimates, but otherwise extra regressors'
                                                 'can be specified in a txt file')

    #parser.add_argument('subject_number', type=int, help='subject number')
    #parser.add_argument('run_number', type=int, help='run number [1, 11]')
    parser.add_argument('--derivs', action='store_true',
                        default=False, help='whether to use the derivatives as regressors')
    parser.add_argument('--noise_estimates', action='store_true',
                        default=False, help='whether to use noise estimates, as of now hardcoded to '
                                            'qa/noisecomp')
    parser.add_argument('--output-model', action='store_true',
                        default=False, help='whether to store the full model (doubles space because of residuals)')
    parser.add_argument('job_number', type=int,
        help='job number for condor submission. it will figure out what run and subject it is')
    parsed = parser.parse_args(sys.argv[1:])
    subject_number = int(parsed.job_number / 11) + 1
    run_number = int(parsed.job_number % 11) + 1

    main(subject_number, run_number, parsed.derivs, parsed.noise_estimates,
         parsed.output_model)
