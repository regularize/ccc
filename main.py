import pickle

import warnings
#warnings.filterwarnings("error")

import os
import re

import torch
torch.set_grad_enabled(False)


from definitions import DATA_DIR, TMP_DIR, RUNS_DIR, OUT_DIR

from ccci.experiments import CCCInstances


def run(model_class, model_name, steps_list, mode, method_agg):

    run_name = (
        model_name + '_' 
        + mode + '_' 
        + method_agg
    )
    runs_path = os.path.join(RUNS_DIR, model_name)
    figs_path = os.path.join(OUT_DIR, 'figs__' + run_name)

    #
    if model_class == 'pythia':
        from dataloaders.dataloader_pythia import DataLoaderPythia
        dataloader_class = DataLoaderPythia
    elif model_class == 'olmo':
        from dataloaders.dataloader_olmo import DataLoaderOLMo
        dataloader_class = DataLoaderOLMo
    else:
        print('model_class {0} not recognized.'.format(model_class))
        return

    # 
    assert mode in ['full', 'compact']

    # 
    C = dataloader_class(model_name, steps_list)
    
    # prepare the 3d data:
    C.prepare()

    # various demonstrations and comparisons: 

    filename = os.path.join(runs_path, run_name + '.pt') ## saved in .save() within experiments.py
    if os.path.isfile(filename):

        objects = []
        with (open(filename, "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break
        print('successfully loaded ', filename)
        ccc_instances = objects[0]
        ccc_instances.figs_path = figs_path
        ccc_instances.runs_path = runs_path
        print(ccc_instances)

        if not os.path.exists(ccc_instances.figs_path):
            os.makedirs(ccc_instances.figs_path)

        ccc_instances.viz_all(get_list=False)

    else:

        print('Coud not load ', filename)

        ccc_instances = CCCInstances(
            C.model_name,
            C.steps_list,
            C.traj_info_nd[mode],
            method_agg=method_agg, 
            runs_path=runs_path,
            run_name=run_name,
            figs_path=figs_path,
            n_head=C.model_args['n_head']
            )
        print(ccc_instances)

        # ccc_instances.demo()
        ccc_instances.viz_all(get_list=True)
        ccc_instances.save()


    # optimize_comp_groups(ccc_instances)

    return


def optimize_comp_groups(ccc_instances):
    """optimizing the score to find optimal components classification"""
    ccc_instances.find_components_grp(
        coords_partitioning='MAD-every30-lincomb',  #'MAD-union-lincomb',  #'MAD-1median-2dif',#
        num_grid_pnts=100,
        max_iters=10000
        )
    return


def main():

    if 1==0:
        model_class = 'pythia'
        model_name = 'pythia-70m-deduped'#  'pythia-160m'#   'pythia-1b'#
        steps_list = [0] + [2 ** i for i in range(10)] + [i * 1000 for i in range(1, 144)]

    if 1==1:
        model_class = 'olmo'
        model_name = 'OLMo-1B'#  'OLMo-7B'#   'OLMo-7B-Twin-2T'#
        steps_list = []
        fname = os.path.join(DATA_DIR, 'OLMo', model_name + '-revisions.txt')
        print(fname)
        with open(fname) as f:
            for line in f.readlines():
                steps_list += [int(re.search('step(\d+)-tokens', line.strip()).group(1))]
        print('steps included in the revision file:', steps_list)

        steps_list = [s for i, s in enumerate(steps_list) if s % 10000 == 0]

    mode = 'full'# 'compact'#
    method_agg = 'lengths-weighted'#  'unweighted'#

    print(steps_list)

    run(model_class, model_name, steps_list, mode, method_agg)
    return

if __name__ == '__main__':
    main()
