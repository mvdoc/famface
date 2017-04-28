function run_tfce_mvdoc_fx(anatype, condition, fnname, nperms, niter, h0_mean)
% example:
% run_tfce_mvdoc_fx('slmskz5vx_svm', 'familiar_vs_unfamilliar-id', 'sl_map_cerebmask.mat', 35, 10000, 0.5)

addpath('~/toolbox/cosmomvpa/mvpa/');
cosmo_set_path();
%%
basedir = '/data/famface/openfmri/results/l1ants_final/model001/task001/';
%sub_ana = 'sub%03d/mvpa/slmskz5vx_svm/%s';
sub_ana = ['sub%03d/mvpa/', anatype, '/%s'];
nsubj = 33;
%nperms = 35;
%condition = 'familiar_vs_unfamiliar-id';

% load mask

%a_mapper = load('/data/famface/openfmri/results/l2ants_final/model001/task001/subjects_all/mask/union_mask_33sbjs_80p.mat');
%mask2 = load('/data/famface/openfmri/results/l2ants_final/model001/task001/subjects_all/mask/union_mask_33sbjs_80p_correct.mat');
%mask2.a = cosmo_structjoin(a_mapper.a, mask2.a);
%mask = mask2;

% fix base-0 to base-1
%mask.fa.i = double(mask.fa.i + 1);
%mask.fa.j = double(mask.fa.j + 1);
%mask.fa.k = double(mask.fa.k + 1);

%% get the original acc maps
orig_ds = cell(1, nsubj);
for isubj = 1:nsubj
    fprintf('Loading sub%03d\n', isubj);
    fn = fullfile(basedir, sprintf(sub_ana, isubj, condition), fnname);
    %tmp = load(fn);
    tmp = cosmo_fmri_dataset(fn);
    %tmp.fa = mask.fa; %cosmo_structjoin(tmp.fa, mask.fa);
    %tmp.a = mask.a;
    %try
    %    tmp.sa = rmfield(tmp.sa, 'cvfolds');
    %catch
    %end
    nsamples = size(tmp.samples, 1);
    if nsamples > 1
        tmp.sa.coefs = (1:nsamples)';
    end
    tmp.sa.chunks = isubj * ones([nsamples, 1]);
    tmp.sa.targets = ones([nsamples, 1]);
    orig_ds{isubj} = tmp;
end

orig_ds = cosmo_stack(orig_ds);

%% get the permuted acc maps
fnname_perm = strrep(fnname, '.mat', '_perm%03d.mat');
if nperms > 0
    perm_ds = cell(1, nperms);

    for ip = 1:nperms
       tmp_dsperm = cell(1, nsubj);
       for isubj = 1:nsubj
          fprintf('sub%03d perm%03d\n', isubj, ip);
          fn = fullfile(basedir, sprintf(sub_ana, isubj, condition), sprintf(fnname_perm, ip));
          %tmp = load(fn);
          tmp = cosmo_fmri_dataset(fn);
          %try
          %  tmp.sa = rmfield(tmp.sa, 'cvfolds');
          %catch
          %end
          %tmp.fa = mask.fa; %cosmo_structjoin(tmp.fa, mask.fa);
          %tmp.a = mask.a;
          nsamples = size(tmp.samples, 1);
          if nsamples > 1
            tmp.sa.coefs = (1:nsamples)';
          end
          tmp.sa.chunks = isubj * ones([nsamples, 1]);
          tmp.sa.targets = ones([nsamples, 1]);
          tmp_dsperm{isubj} = tmp;
       end
       perm_ds{ip} = cosmo_stack(tmp_dsperm);
    end
end
%%  neighbors
nbr = cosmo_cluster_neighborhood(orig_ds);%load('/data/famface/openfmri/results/l2ants_final/model001/task001/subjects_all/mask/cosmo_union_mask_33sbjs_80p_neighbors.mat');

opt = struct();
opt.cluster_stat='tfce';
opt.niter=niter;
opt.h0_mean=h0_mean;
opt.seed=42;
if nperms > 0
    opt.null=perm_ds;
end

dirout = sprintf('/data/famface/openfmri/results/l2ants_final/model001/task001/subjects_all/stats/mvpa/%s/%s', anatype, condition);
if ~exist(dirout, 'dir')
    mkdir(dirout)
end
if nperms > 0
    nulltype = 'perms';
else
    nulltype = 'flip';
end

if nsamples > 1
    for coef = unique(orig_ds.sa.coefs)'
        orig_ds_ = cosmo_slice(orig_ds, orig_ds.sa.coefs == coef);
        if nperms > 0
            perm_ds_ = cellfun(@(x) cosmo_slice(x, x.sa.coefs == coef), perm_ds, 'UniformOutput', false);
            opt.null=perm_ds_;
        end
        fprintf('Running for coef %d\n', coef);
        z_ds = cosmo_montecarlo_cluster_stat(orig_ds_, nbr, opt);
        
           
        fn_ = strrep(fnname, '.mat', '_tfce_%s_b%d_coef%02d.mat');
        fn = sprintf(fn_, nulltype, niter, coef);
        fnout = fullfile(dirout, fn);
        try
            save(fnout, '-struct', 'z_ds');
            cosmo_map2fmri(z_ds, strrep(fnout, '.mat', '.nii'));
        catch
            fprintf('Couldnt save in %s, saving locally\n', fnout)
            fnout = [anatype, '+', condition, fn];
            save(fnout, '-struct', 'z_ds');
            cosmo_map2fmri(z_ds, strrep(fnout, '.mat', '.nii'));
        end
    end
else
    z_ds = cosmo_montecarlo_cluster_stat(orig_ds, nbr, opt);

   
    fn_ = strrep(fnname, '.mat', '_tfce_%s_b%d.mat');
    fn = sprintf(fn_, nulltype, niter);
    fnout = fullfile(dirout, fn);

    try
        save(fnout, '-struct', 'z_ds');
        cosmo_map2fmri(z_ds, strrep(fnout, '.mat', '.nii'));
    catch
        fprintf('Couldnt save in %s, saving locally\n', fnout)
        fnout = [anatype, '+', condition, fn];
        save(fnout, '-struct', 'z_ds');
        cosmo_map2fmri(z_ds, strrep(fnout, '.mat', '.nii'));
    end
end
end
