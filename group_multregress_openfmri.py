"""
ROUGH DRAFT

for help:
contact: annepark@mit.edu
"""

import os
from nipype import config
#config.enable_provenance()

from nipype import Workflow, Node, MapNode, Function
from nipype import DataGrabber, DataSink
from nipype.interfaces.fsl import (Merge, FLAMEO, ContrastMgr,
                                   SmoothEstimate, Cluster, ImageMaths, MultipleRegressDesign)
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
from nipype.interfaces.fsl.maths import BinaryMaths

get_len = lambda x: len(x)

def l1_contrasts_num(model_id, task_id, dataset_dir):
    import numpy as np
    import os
    contrast_def = []
    contrasts = 0
    contrast_file = os.path.join(dataset_dir, 'models', 'model%03d' % model_id,
                                 'task_contrasts.txt')
    if os.path.exists(contrast_file):
        with open(contrast_file, 'rt') as fp:
            contrast_def.extend([np.array(row.split()) for row in fp.readlines() if row.strip()])
    for row in contrast_def:
        if row[0] != 'task%03d' % task_id:
            continue
        contrasts = contrasts + 1
    cope_id = range(1, contrasts + 1)
    return cope_id


def get_sub_vars(dataset_dir, task_id, model_id):
    import numpy as np
    import os
    import pandas as pd
    sub_list_file = os.path.join(dataset_dir, 'groups', 'participant_key.txt')
    behav_file = os.path.join(dataset_dir, 'groups', 'behav.txt')
    group_contrast_file = os.path.join(dataset_dir, 'groups', 'contrasts.txt')

    subs_list = pd.read_table(sub_list_file, index_col=0)['task%03d' % int(task_id)]
    subs_needed = subs_list.index[np.nonzero(subs_list)[0]]
    behav_info = pd.read_table(behav_file, index_col=0)

    missing_subjects = np.setdiff1d(subs_needed, behav_info.index.tolist())
    if len(missing_subjects) > 0:
        raise ValueError('Subjects %s are missing from participant key' % ' '.join(missing_subjects))

    contrast_defs=[]
    with open(group_contrast_file, 'rt') as fp:
        contrast_defs = fp.readlines()

    contrasts = []
    for row in contrast_defs:
        if 'task%03d' % int(task_id) not in row:
            continue
        regressor_names = eval('[' + row.split(' [')[1].split(']')[0] + ']')
        for val in regressor_names:
            if val not in behav_info.keys():
                raise ValueError('Regressor %s not in behav.txt file' % val)
        contrast_name = row.split()[1]
        contrast_vector = np.array(row.split('] ')[1].rstrip().split()).astype(float).tolist()
        con = [tuple([contrast_name, 'T', regressor_names, contrast_vector])]
        contrasts.append(con)
    
    regressors_needed = []
    for idx, con in enumerate(contrasts):
        model_regressor = {}
        for cond in con[0][2]:
            values = behav_info.ix[subs_needed, cond].values
            if tuple(np.unique(values).tolist()) not in [(1,), (0, 1)]:
                values = values - values.mean()
            model_regressor[cond] = values.tolist()
        regressors_needed.append(model_regressor)
    groups = [1 for val in subs_needed]
    return regressors_needed, contrasts, groups, subs_needed.values.tolist()


def run_palm(cope_file, design_file, contrast_file, group_file, mask_file, 
             cluster_threshold=3.09):
    import os
    from glob import glob
    from nipype.interfaces.base import CommandLine
    #cmd = ("palm -i {cope_file} -m {mask_file} -d {design_file} -t {contrast_file} -eb {group_file} -T " 
    #       "-C {cluster_threshold} -Cstat extent -fdr -noniiclass -twotail -logp -zstat")
    #cl = CommandLine(cmd.format(cope_file=cope_file, mask_file=mask_file, design_file=design_file, 
    #                            contrast_file=contrast_file,
    #                            group_file=group_file, cluster_threshold=cluster_threshold))

    # XXX: ideally we should make it more fancy, but since we're only doing
    # 1-sample t-tests we need to omit the design, contrast, and group files
    # as for PALM's FAQs
    cmd = ("palm -i {cope_file} -m {mask_file} -T "
           "-C {cluster_threshold} -Cstat extent -fdr -noniiclass -twotail -logp -zstat")
    cl = CommandLine(cmd.format(cope_file=cope_file, mask_file=mask_file,
                                cluster_threshold=cluster_threshold))
    results = cl.run(terminal_output='file')
    return [os.path.join(os.getcwd(), val) for val in sorted(glob('palm*'))]


def group_multregress_openfmri(dataset_dir, model_id=None, task_id=None, l1output_dir=None, out_dir=None, 
                               no_reversal=False, plugin=None, plugin_args=None, flamemodel='flame1',
                               nonparametric=False, use_spm=False):

    meta_workflow = Workflow(name='mult_regress')
    meta_workflow.base_dir = work_dir
    for task in task_id:
        cope_ids = l1_contrasts_num(model_id, task, dataset_dir)
        regressors_needed, contrasts, groups, subj_list = get_sub_vars(dataset_dir, task, model_id)
        for idx, contrast in enumerate(contrasts):
            wk = Workflow(name='model_%03d_task_%03d_contrast_%s' % (model_id, task, contrast[0][0]))

            info = Node(util.IdentityInterface(fields=['model_id', 'task_id', 'dataset_dir', 'subj_list']),
                        name='infosource')
            info.inputs.model_id = model_id
            info.inputs.task_id = task
            info.inputs.dataset_dir = dataset_dir
            
            dg = Node(DataGrabber(infields=['model_id', 'task_id', 'cope_id'],
                                  outfields=['copes', 'varcopes']), name='grabber')
            dg.inputs.template = os.path.join(l1output_dir,
                                              'model%03d/task%03d/%s/%scopes/%smni/%scope%02d.nii%s')
            if use_spm:
                dg.inputs.template_args['copes'] = [['model_id', 'task_id', subj_list, '', 'spm/',
                                                     '', 'cope_id', '']]
                dg.inputs.template_args['varcopes'] = [['model_id', 'task_id', subj_list, 'var', 'spm/',
                                                        'var', 'cope_id', '.gz']]
            else:
                dg.inputs.template_args['copes'] = [['model_id', 'task_id', subj_list, '', '', '', 
                                                     'cope_id', '.gz']]
                dg.inputs.template_args['varcopes'] = [['model_id', 'task_id', subj_list, 'var', '',
                                                        'var', 'cope_id', '.gz']]
            dg.iterables=('cope_id', cope_ids)
            dg.inputs.sort_filelist = False

            wk.connect(info, 'model_id', dg, 'model_id')
            wk.connect(info, 'task_id', dg, 'task_id')

            model = Node(MultipleRegressDesign(), name='l2model')
            model.inputs.groups = groups
            model.inputs.contrasts = contrasts[idx]
            model.inputs.regressors = regressors_needed[idx]
            
            mergecopes = Node(Merge(dimension='t'), name='merge_copes')
            wk.connect(dg, 'copes', mergecopes, 'in_files')
            
            mergevarcopes = Node(Merge(dimension='t'), name='merge_varcopes')
            wk.connect(dg, 'varcopes', mergevarcopes, 'in_files')
            
            mask_file = fsl.Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')
            flame = Node(FLAMEO(), name='flameo')
            flame.inputs.mask_file =  mask_file
            flame.inputs.run_mode = flamemodel
            #flame.inputs.infer_outliers = True

            wk.connect(model, 'design_mat', flame, 'design_file')
            wk.connect(model, 'design_con', flame, 't_con_file')
            wk.connect(mergecopes, 'merged_file', flame, 'cope_file')
            wk.connect(mergevarcopes, 'merged_file', flame, 'var_cope_file')
            wk.connect(model, 'design_grp', flame, 'cov_split_file')
            
            if nonparametric:
                palm = Node(Function(input_names=['cope_file', 'design_file', 'contrast_file', 
                                                  'group_file', 'mask_file', 'cluster_threshold'],
                                     output_names=['palm_outputs'],
                                     function=run_palm),
                            name='palm')
                palm.inputs.cluster_threshold = 3.09
                palm.inputs.mask_file = mask_file
                palm.plugin_args = {'sbatch_args': '-p om_all_nodes -N1 -c2 --mem=10G', 'overwrite': True}
                wk.connect(model, 'design_mat', palm, 'design_file')
                wk.connect(model, 'design_con', palm, 'contrast_file')
                wk.connect(mergecopes, 'merged_file', palm, 'cope_file')
                wk.connect(model, 'design_grp', palm, 'group_file')
                
            smoothest = Node(SmoothEstimate(), name='smooth_estimate')
            wk.connect(flame, 'zstats', smoothest, 'zstat_file')
            smoothest.inputs.mask_file = mask_file
        
            cluster = Node(Cluster(), name='cluster')
            wk.connect(smoothest,'dlh', cluster, 'dlh')
            wk.connect(smoothest, 'volume', cluster, 'volume')
            cluster.inputs.connectivity = 26
            cluster.inputs.threshold = 2.3
            cluster.inputs.pthreshold = 0.05
            cluster.inputs.out_threshold_file = True
            cluster.inputs.out_index_file = True
            cluster.inputs.out_localmax_txt_file = True
            
            wk.connect(flame, 'zstats', cluster, 'in_file')
    
            ztopval = Node(ImageMaths(op_string='-ztop', suffix='_pval'),
                           name='z2pval')
            wk.connect(flame, 'zstats', ztopval,'in_file')
            
            sinker = Node(DataSink(), name='sinker')
            sinker.inputs.base_directory = os.path.join(out_dir, 'task%03d' % task, contrast[0][0])
            sinker.inputs.substitutions = [('_cope_id', 'contrast'),
                                           ('_maths_', '_reversed_')]
            
            wk.connect(flame, 'zstats', sinker, 'stats')
            wk.connect(cluster, 'threshold_file', sinker, 'stats.@thr')
            wk.connect(cluster, 'index_file', sinker, 'stats.@index')
            wk.connect(cluster, 'localmax_txt_file', sinker, 'stats.@localmax')
            if nonparametric:
                wk.connect(palm, 'palm_outputs', sinker, 'stats.palm')

            if not no_reversal:
                zstats_reverse = Node( BinaryMaths()  , name='zstats_reverse')
                zstats_reverse.inputs.operation = 'mul'
                zstats_reverse.inputs.operand_value = -1
                wk.connect(flame, 'zstats', zstats_reverse, 'in_file')
                
                cluster2=cluster.clone(name='cluster2')
                wk.connect(smoothest, 'dlh', cluster2, 'dlh')
                wk.connect(smoothest, 'volume', cluster2, 'volume')
                wk.connect(zstats_reverse, 'out_file', cluster2, 'in_file')
                
                ztopval2 = ztopval.clone(name='ztopval2')
                wk.connect(zstats_reverse, 'out_file', ztopval2, 'in_file')
                
                wk.connect(zstats_reverse, 'out_file', sinker, 'stats.@neg')
                wk.connect(cluster2, 'threshold_file', sinker, 'stats.@neg_thr')
                wk.connect(cluster2, 'index_file',sinker, 'stats.@neg_index')
                wk.connect(cluster2, 'localmax_txt_file', sinker, 'stats.@neg_localmax')
            meta_workflow.add_nodes([wk])
    return meta_workflow

if __name__ == '__main__':
    import argparse
    defstr = ' (default %(default)s)'
    parser = argparse.ArgumentParser(prog='group_multregress_openfmri.py',
                                     description=__doc__)
    parser.add_argument('-m', '--model', default=1, type=int,
                        help="Model index" + defstr)
    parser.add_argument('-t', '--task', default=[1], nargs='+',
                        type=int, help="Task index" + defstr)
    parser.add_argument("-o", "--output_dir", dest="outdir",
                        help="Output directory base")
    parser.add_argument('-d', '--datasetdir', required=True)
    parser.add_argument("-l1", "--l1_output_dir", dest="l1out_dir",
                        help="l1_output directory ")
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="Output directory base")
    parser.add_argument("-p", "--plugin", dest="plugin",
                        default='Linear',
                        help="Plugin to use" + defstr)
    parser.add_argument("--plugin_args", dest="plugin_args",
                        help="Plugin arguments")
    parser.add_argument("--norev",action='store_true',
                        help="do not generate reverse contrasts")
    parser.add_argument("--use_spm",action='store_true', default=False,
                        help="use spm estimation results from 1st level")
    parser.add_argument("--nonparametric", action='store_true', default=False,
                        help="Run non-parametric estimation using palm" + defstr)
    parser.add_argument('-f','--flame', dest='flamemodel', default='flame1',
                        choices=('ols', 'flame1', 'flame12'),
                        help='tool to use for dicom conversion' + defstr)
    parser.add_argument("--sleep", dest="sleep", default=60., type=float,
                        help="Time to sleep between polls" + defstr)
    parser.add_argument("--write-graph", default="",
                        help="Do not run, just write the graph to specified file")
    args = parser.parse_args()
    outdir = args.outdir
    work_dir = os.getcwd()

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    if args.outdir:
        outdir = os.path.abspath(outdir)
    if args.l1out_dir:
        l1_outdir=os.path.abspath(args.l1out_dir)
    else:
        l1_outdir=os.path.join(args.datasetdir, 'l1output')

    outdir = os.path.join(outdir, 'model%03d' % args.model)

    wf = group_multregress_openfmri(model_id=args.model,
                                    task_id=args.task,
                                    l1output_dir=l1_outdir,
                                    out_dir=outdir,
                                    dataset_dir=os.path.abspath(args.datasetdir),
                                    no_reversal=args.norev,
                                    flamemodel=args.flamemodel,
                                    nonparametric=args.nonparametric,
                                    use_spm=args.use_spm)
    wf.config['execution']['poll_sleep_duration'] = args.sleep

    if args.write_graph:
        wf.write_graph(args.write_graph, graph2use='orig')
    elif args.plugin_args:
        wf.run(args.plugin, plugin_args=eval(args.plugin_args))
    else:
        wf.run(args.plugin)
