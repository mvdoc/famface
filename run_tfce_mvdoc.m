addpath('~/toolbox/cosmomvpa/mvpa/');
cosmo_set_path();
%%
basedir = '/data/famface/openfmri/results/l1ants_final/model001/task001/';
sub_ana = 'sub%03d/mvpa/slmskz5vx_svm/%s';

nsubj = 33;
nperms = 35;
condition = 'familiar_vs_unfamiliar-id';

% load mask

a_mapper = load('/data/famface/openfmri/results/l2ants_final/model001/task001/subjects_all/mask/union_mask_33sbjs_80p.mat');
mask2 = load('/data/famface/openfmri/results/l2ants_final/model001/task001/subjects_all/mask/union_mask_33sbjs_80p_correct.mat');
mask2.a = cosmo_structjoin(a_mapper.a, mask2.a);
mask = mask2;

mask.fa.i = double(mask.fa.i);
mask.fa.j = double(mask.fa.j);
mask.fa.k = double(mask.fa.k);

%% get the original acc maps
orig_ds = cell(1, nsubj);
for isubj = 1:nsubj
    fprintf('sub%03d\n', isubj);
    tmp = load(fullfile(basedir, sprintf(sub_ana, isubj, condition), 'sl_map.mat'));
    tmp.fa = cosmo_structjoin(tmp.fa, mask.fa);
    tmp.a = mask.a;
    tmp.sa = rmfield(tmp.sa, 'cvfolds');
    tmp.sa.chunks = isubj;
    tmp.sa.targets = 1;
    orig_ds{isubj} = tmp;
end

orig_ds = cosmo_stack(orig_ds);

%% get the permuted acc maps
perm_ds = cell(1, nperms);

for ip = 1:nperms
   tmp_dsperm = cell(1, nsubj);
   for isubj = 1:nsubj
      fprintf('sub%03d perm%03d\n', isubj, ip);
      tmp = load(fullfile(basedir, sprintf(sub_ana, isubj, condition), sprintf('sl_map_perm%03d.mat', ip)));
      tmp.sa = rmfield(tmp.sa, 'cvfolds');
      tmp.fa = cosmo_structjoin(tmp.fa, mask.fa);
      tmp.a = mask.a;
      tmp.sa.chunks = isubj;
      tmp.sa.targets = 1;
      tmp_dsperm{isubj} = tmp;
   end
   perm_ds{ip} = cosmo_stack(tmp_dsperm);
end

%% load neighbors
nbr = cosmo_cluster_neighborhood(mask);%load('/data/famface/openfmri/results/l2ants_final/model001/task001/subjects_all/mask/cosmo_union_mask_33sbjs_80p_neighbors.mat');

opt = struct();
opt.cluster_stat='tfce';
opt.niter=10000;
opt.h0_mean=.50;
opt.seed=42;
opt.null = perm_ds;

z_ds = cosmo_montecarlo_cluster_stat(orig_ds, nbr, opt);

dirout = '/data/famface/openfmri/results/l2ants_final/model001/task001/subjects_all/stats/mvpa/slmskz5vx_svm/familiar_vs_unfamiliar-id';
fnout = fullfile(dirout, 'sl_map_tfce_perms_b10000.mat');
%fnout = fullfile(dirout, 'sl_map_tfce_flip_b1000.mat');
save(fnout, '-struct', 'z_ds');
cosmo_map2fmri(z_ds, strrep(fnout, '.mat', '.nii'));