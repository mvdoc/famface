# TITLE

This repository contains preprocessing and analysis scripts for "..." by
Matteo Visconti di Oleggio Castello, Yaroslav O. Halchenko, J. Swaroop
Guntupalli, Jason D. Gors, and M. Ida Gobbini.

## Preprocessing and GLM modeling

- `fmri_ants_openfmri.py`: nipype pipeline to perform preprocessing
  (spatial normalization to MNI 2 mm using ANTs, first and second level
univariate analysis using FSL)
- `pymvpa_hrf.py`: script to run a GLM using PyMVPA and Nipy, to extract
  betas used for multivariate analysis
- `make_unionmask.py`: script to make a union mask
- `stack_betas.py`: script to stack betas for multivariate analysis

## Analysis

- `group_multregress_openfmri.py`: nipype pipeline to perform third
  (group) level univariate analysis with FSL
- `run_sl.py`: main script to run searchlight analyses
- `pymvpa2cosmo.py`: script to convert PyMVPA datasets into CoSMoMVPA
  datasets for statistical testing using TFCE
- `run_tfce_mvdoc_fx.m`: script to run TFCE on accuracy maps using
  CoSMoMVPA

## Auxiliary modules

- `mds_rois.py`: contains functions to run MDS analyses

