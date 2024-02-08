import numpy as np
import math

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .classification_prep import CCC
from .triclassification import TriClassificationInstances

from .utils import Groups, Experiment, class_str, my_cmap

from dataclasses import replace

import os
import pickle


##########


class CCCInstances(CCC):
    def __init__(
        self,
        model_name,
        steps_list,
        traj_info_nd,
        method_agg,
        runs_path,
        run_name,
        **kwargs):
        """

        Parameters:
        -----------
            method_agg: {'unweighted', 'lengths-weighted'}
                all of the computations stored in self.experiments
                will depend on the choice of method_agg.
            experiments: List[Experiment]
            path: str

        Notes:
        ------
        Considering the following in running the methods
        would help avoid unnecessary computations:
            Method viz_trajectories() uses an instance of class Trajectories2d,
                hence computes the scores to be used as plot title.
            We get this score and store it in self.experiments.
            Therefore, calling .compute_scores() with a similar num_grid_points
                would be fast (in fact, unnecessary).

        In a choice between a typing.NamedTuple and a frozen dataclass,
            we opt for the latter, and we create a new entry in our
            experiments list whenever any experiment data needs modification.
        However, to keep track of the origin for each entry in the list,
            we define some special fields: source, method, detail.

        TODO: to not call .create_trajectories2d(),
            once in .compute_groups_traj2d(),
            once in .viz_trajectories(),
            once in .compute_scores().
        """
        super(CCCInstances, self).__init__(
            model_name,
            steps_list,
            traj_info_nd,
            **kwargs)
        self.method_agg = method_agg
        self.experiments = []
        self.runs_path = runs_path
        self.run_name = run_name

        if 'figs_path' not in kwargs:
            self.save_figs = False
        else:
            self.save_figs = True
            self.figs_path = kwargs['figs_path']
            if not os.path.exists(self.figs_path):
                os.makedirs(self.figs_path)

        print('Instantiated CCCInstances.')

    def __str__(self):
        return class_str(
            self.__class__.__name__, 
            [(attr, getattr(self, attr)) for attr in ['model_name', 'runs_path', 'figs_path']] +
            [('3d tensor', self.mags.shape)]
            )

    def _report_len_experiments(self):
        print('length of current .experiments list: {0}'.format(len(self.experiments)))
        
    def get_list_comps_grps(self, methods_list):
        """Return a list of several component groupings

        Parameters:
        -----------
            methods_list: {'wte', 'random', 'all_size1'}

        Returns:
        --------
            a list of 3-tuples,
                each tuple being a pair (components_grp,components_grp_c),
                and a 'source' of how the grouping has been generated;
                the two component groups are not necessarily disjoint or complementary.
        """

        P_ = []

        if 'wte' in methods_list:
            P_ += [([0], list(range(1, self.n_comps)), 'comps_grps:wte')]

        if 'random' in methods_list:
            for kk in range(2):
                print('aggregating random columns')
                comps_grp1 = np.random.randint(self.n_comps, size=(self.n_comps // 2)).tolist()
                comps_grp1 = [i for i in range(self.n_comps) if i in comps_grp1]  # uniques
                comps_grp2 = [i for i in range(self.n_comps) if i not in comps_grp1]
                P_ += [(comps_grp1, comps_grp2, 'comps_grps:random')]

        if 'all_size1' in methods_list and self.n_comps < 10:
            P_ += [(
                [j],
                [i for i in range(self.n_comps) if i != j],
                'comps_grps:all_size1'
                )
                for i in range(self.n_comps)
                ]

        return P_

    def generate_list_experiments(self):
        """Generate a list of experiments,
        namely a (partial) Groups instance
        and some information (as 'source', 'method', 'detail' fields), 
        and append to the current experiments list."""

        print('----- adding new experiments:')
        self._report_len_experiments()

        if 1 == 1:

            print('--- TriClassificationInstances.plot_list_triclassifications() :')
            ClsInstances = TriClassificationInstances(
                self.mags_outs,
                self.weights_comps
                )
            # a somewhat costly step:
            fig, axs = ClsInstances.plot_list_triclassifications()
            if self.save_figs:
                plt.savefig(os.path.join(self.figs_path, 'ccci_viz_plot_list_triclassifications.png'), dpi=300)
                plt.close(fig)


            if 1 == 1:
                # (kind "TCI") all 4 groups directly from TriClassification:
                self.experiments += ClsInstances.experiments
                self._report_len_experiments()

            if 1 == 1:
                # (kind "TCI:keep_comps_grps")
                # ignoring the optimal inlier/outliers from TCI outputs:
                self.experiments += [
                    replace(
                        exp_,
                        source=getattr(exp_, 'source') + ':keeping_comps_grps',
                        groups=Groups(
                            {
                                k: getattr(exp_, 'groups').groups[k]
                                for k in (getattr(exp_, 'groups').groups.keys()
                                          & {'comps_grp1', 'comps_grp2'})
                                }
                            )
                        )
                    for exp_ in ClsInstances.experiments]
                self._report_len_experiments()

            if 1 == 1:
                # (kind "TCI:keep_coords_in_out")
                # ignoring the optimal inlier/outliers from TCI outputs:
                self.experiments += [
                    replace(
                        exp_,
                        source=getattr(exp_, 'source') + ':keeping_coords_in_out',
                        groups=Groups(
                            {
                                k: getattr(exp_, 'groups').groups[k]
                                for k in (getattr(exp_, 'groups').groups.keys()
                                          & {'coords_in', 'coords_out'})
                                }
                            )
                        )
                    for exp_ in ClsInstances.experiments]
                self._report_len_experiments()

        if 1 == 1:
            # (kind "comps_grps:...")
            # considering a list of comps_grps
            # and generating rest via all possible outlier methods:
            print('----- .get_list_comps_grps() :')
            list_comps_grps = self.get_list_comps_grps(
                methods_list=[
                    'wte',
                    'random',
                    # 'all_size1',
                    ]
                )

            self.experiments += [
                Experiment(
                    source=p[2],
                    method='',
                    detail='',
                    groups=Groups(
                        {'comps_grp1': p[0],
                         'comps_grp2': p[1]}
                        ),
                    scores=(None, None)
                    )
                for p in list_comps_grps
                ]
            self._report_len_experiments()

        return

    def compute_groups_traj2d(self):
        """Complete the missing keys from the Groups."""
        print('------------ completing any incomplete groups field in the experiments list')
        self._report_len_experiments()

        list_method_find_outliers = [
            'MAD-union-lincomb',
            'MAD-union-dif', 'MAD-union-sum',
            'MAD-1median-2dif', 'MAD-2median-1dif',
            'MAD-last-dif', 'MAD-last-sum',
            ]
        list_method_find_grps = ['sort-sum']

        print('list_method_find_outliers:', list_method_find_outliers)
        print('list_method_find_grps    :', list_method_find_grps)
        ##

        new_list = []
        for j, exp_ in enumerate(self.experiments):
            print('---- Experiment {0} out of {1}:'.format(j, len(self.experiments)))
            print(exp_)

            g = getattr(exp_, 'groups')  # an instance of the class Groups

            if g.has_all_keys():
                new_list += [exp_]

            elif g.has_comps_keys() and not g.has_coords_keys():
                for mthd in list_method_find_outliers:
                    Traj2d = self.create_trajectories2d(
                        g.groups['comps_grp1'],
                        g.groups['comps_grp2'],
                        mthd,
                        method_agg=self.method_agg
                        )
                    coords_out = Traj2d.get_coords_out()
                    coords_in = [i for i in range(self.n_embd) if i not in coords_out]
                    new_list += [
                        Experiment(
                            source=getattr(exp_, 'source') + '|find_outliers_traj2d',
                            method=getattr(exp_, 'method') + '|'+mthd,
                            detail='',
                            groups=Groups(
                                {'comps_grp1': g.groups['comps_grp1'],
                                 'comps_grp2': g.groups['comps_grp2'],
                                 'coords_out': coords_out,
                                 'coords_in': coords_in}
                                ),
                            scores=(None, None)
                            )
                        ]
                    print(new_list[-1], '\n')

            elif g.has_coords_keys():
                for mthd in list_method_find_grps:
                    pass
                    #?? # need access to a mags_outs

                    #print(new_list[-1], '\n')

        self.experiments = new_list

        return

    def sort_experiments(self):
        """Sort the experiments list
        based on the last letters of the 'method' field.
        """
        print('\n\n------------ sorting self.experiments\n\n')
        idx = np.argsort([
            getattr(exp_, 'method')[::-1]
            for exp_ in self.experiments
            ])
        self.experiments = [self.experiments[i] for i in idx]
        return

    def compute_scores(self, num_grid_points):
        """Compute and store the scores
        for each instance of Experiment in self.experiments,
        if the 'scores' field (a tuple) has any None values.
        """
        print('----- .compute_scores for the current experiments list')
        self._report_len_experiments()

        self.experiments = [
            replace(
                exp_,
                scores=self.create_trajectories2d(
                    (g:=getattr(exp_, 'groups').groups)['comps_grp1'], g['comps_grp2'],
                    {'coords_in': g['coords_in'], 'coords_out': g['coords_out']},
                    method_agg=self.method_agg
                    ).get_scores(num_grid_points)
                )
            if any(map(lambda x: x is None, getattr(exp_, 'scores')))
            else exp_
            for exp_ in self.experiments
            ]
        return

    ############

    def viz_agg1d(self):
        """Plot 1-d aggregation of all components."""
        print('\n\n------------ plot 1-d aggregation of all')
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(np.sum(self.mags, axis=-1).T)
        ax.set_xlabel(
            'steps ({0}...{1})'.format(
                min(self.steps_list),
                max(self.steps_list))
            )

        if self.save_figs:
            plt.savefig(os.path.join(self.figs_path, 'ccci_viz_agg1d.png'), dpi=300)
            plt.close(fig)

        return

    def viz_single_grp(self, ind_comp):
        """Plot only a given component."""
        assert 0 <= ind_comp and ind_comp < self.n_comps
        print('\n\n------------ plot only comp {0}'.format(ind_comp))
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot(self.mags[:, :, ind_comp].T)
        ax.set_xlabel('steps ({0}...{1})'.format(
            min(self.steps_list),
            max(self.steps_list)))

        if self.save_figs:
            plt.savefig(os.path.join(self.figs_path, 'ccci_viz_single_grp.png'), dpi=300)
            plt.close(fig)

        return

    def viz_scores(self):
        """Visualize the scores for all experiments on a 2d plot,
        where colors indicate the source of the experiment."""
        print('\n\n------------ running viz_scores()')
        self._report_len_experiments()
        if len(self.experiments) > 0:

            scores = np.array([
                getattr(exp_, 'scores')
                for exp_ in self.experiments
                ])
            sources = [
                getattr(exp_, 'source')
                for exp_ in self.experiments
                ]
            sources = np.unique(sources, return_inverse=True)

            assert scores.shape == (len(self.experiments), 2)

            fig, ax = plt.subplots(1, 1)
            sct = ax.scatter(
                scores[:, 0], scores[:, 1],
                c=sources[1], cmap=plt.get_cmap('tab20')
                )
            handles, labels = sct.legend_elements(prop="colors", alpha=1.0)
            lgd = ax.legend(
                handles, sources[0],
                loc="center left", 
                title="experiment type",
                bbox_to_anchor=(1, 0.5), 
                )
           
            ax.set(
                # xlim=(0,None), ylim=(0,None),
                # xscale='log',
                title='scores',
                xlabel='irregularity, of the 2d trajectory plot of inliers',
                ylabel='fraction of outliers')
            '''
            for i in range(len(self.experiments)):
                ax.annotate(
                    i, (
                        scores[i, 0] + np.random.randn(1).item() * 0.02, 
                        scores[i, 1] + np.random.randn(1).item() * 0.02
                        )
                        )
            '''
            if self.save_figs:
                plt.savefig(
                    os.path.join(self.figs_path, 'ccci_viz_scores.png'), 
                    bbox_extra_artists=(lgd,), bbox_inches='tight',
                    dpi=300)
                plt.close(fig)

        return

    def viz_trajectories(self):
        """Plot the 2d trajectories for each experiment:
        A total of four subplots:
            subplot (1,1): trajectories for inlier coordinates; 
                and the mean trajectory;
            subplot (1,2): trajectories for outlier coordinates; 
                and the mean trajectory for inliers;
                and the bounding box for subplot (1,1);
            subplot (2,1): quiver plot for average displacements of
                all inlier trajectories;
            subplot (2,2): an illustration of components groups memberships.
        """
        print('\n\n------------ running viz_trajectories()')
        self._report_len_experiments()
        if len(self.experiments) > 0:

            new_list = []
            for ii, exp_ in enumerate(self.experiments):
                print(
                    '---- plot and report; experiment {0} of {1} '.format(
                        ii, len(self.experiments))
                    )
                print(exp_)

                gr = getattr(exp_, 'groups').groups

                Traj2d = self.create_trajectories2d(
                    gr['comps_grp1'],
                    gr['comps_grp2'],
                    {
                        'coords_in': gr['coords_in'],
                        'coords_out': gr['coords_out']
                        },
                    method_agg=self.method_agg
                    )
                
                scores, fig, axs = Traj2d.plot(
                    len_filter=10,
                    plot_quiver=True,
                    num_grid_points_quiver_plot=100
                    )

                new_list += [replace(exp_, scores=(scores[0], scores[1]))]
                print(new_list[-1])

                '''we have stored the scores in self.experiments 
                even though we have deleted the Traj2d instance created back then. 
                So we will not call Traj2d.get_scores()
                but Traj2d.plot() calls Traj2d.get_scores() for the title of the plot.....
                '''

                self.viz_comps_grps(axs[3], gr['comps_grp1'], gr['comps_grp2'])

                fig.suptitle(
                    ' // '.join(
                        [getattr(exp_, 'source'),
                         getattr(exp_, 'method'),
                         getattr(exp_, 'detail')
                         ]
                        )
                    )
                if self.save_figs:
                    plt.savefig(os.path.join(self.figs_path, 'ccci_viz_trajectories_{0}.png'.format(ii)), dpi=300)
                    plt.close(fig)

                self._report_len_experiments()
                print('---- end of plot and report.')
            self.experiments = new_list
        return

    def _plot_mat_cov_gram(self, inds, titl):
        """Visualize a given matrix 'inds',
        as well as the matrices inds@inds.T and inds.T@inds,
        after permuting the columns of inds according to their sums.
        """
        idx = np.argsort(np.sum(inds, axis=0))
        inds = inds[:, idx]

        fig, ax = plt.subplots(1, 1)
        sns.heatmap(inds, ax=ax, cmap=my_cmap, center=0)
        ax.set_title(
            '(permuted coords) {0} indicators identified by {1} experiments:'.format(
                titl, len(self.experiments))
            )
        if self.save_figs:
            plt.savefig(os.path.join(self.figs_path, 'ccci_viz_' + titl + '_indic.png'), dpi=300)
            plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        sns.heatmap(np.matmul(inds, inds.T), ax=ax, cmap=my_cmap, center=0)
        ax.set_title(
            'Cov matrix of {0} indicators identified by {1} experiments:'.format(
                titl, len(self.experiments))
            )
        if self.save_figs:
            plt.savefig(os.path.join(self.figs_path, 'ccci_viz_' + titl + '_cov.png'), dpi=300)
            plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        sns.heatmap(np.matmul(inds.T, inds), ax=ax, cmap=my_cmap, center=0)
        ax.set_title(
            '(permuted) Gram matrix of {0} indicators identified by {1} experiments:'.format(
                titl, len(self.experiments))
            )
        if self.save_figs:
            plt.savefig(os.path.join(self.figs_path, 'ccci_viz_' + titl + '_gram.png'), dpi=300)
            plt.close(fig)

    def viz_outputs_outliers(self):
        """Visualize outliers/inliers memberships across all experiments."""
        print('\n\n------------ running viz_outputs_outliers()')
        self._report_len_experiments()
        if len(self.experiments) > 0:
            inds = -1.0 + np.zeros((len(self.experiments), self.n_embd))
            for i, exp_ in enumerate(self.experiments):
                gr = getattr(exp_, 'groups').groups
                if 'coords_out' in gr:
                    inds[i, gr['coords_out']] = 1.0
        self._plot_mat_cov_gram(inds, 'outliers')
        self._report_len_experiments()
        print('\n\n------------ finished viz_outputs_outliers()')
        return

    def viz_outputs_comps(self):
        """Visualize componenets groups memberships across all experiments."""
        print('\n\n------------ running viz_outputs_comps()')
        self._report_len_experiments()
        if len(self.experiments) > 0:
            inds = np.zeros((len(self.experiments), self.n_comps))
            for i, exp_ in enumerate(self.experiments):
                gr = getattr(exp_, 'groups').groups
                if 'comps_grp1' in gr and 'comps_grp2' in gr:
                    gs = [gr['comps_grp1'], gr['comps_grp2']]
                    #print(gs)
                    ii = [int(0 in grp) for grp in gs]
                    #print(ii)
                    gs = [gs[j] for j in np.argsort(ii).tolist()]
                    #print(gs)
                    inds[i, gs[0]] = 1.0
                    inds[i, gs[1]] = -1.0
                    #print(inds[i, :])
        self._plot_mat_cov_gram(inds, 'components')
        self._report_len_experiments()
        print('\n\n------------ finished viz_outputs_comps()')
        return

    def viz_comps_grps(self, ax, comps_grp1, comps_grp2):
        """Visualize components groups memberships for an experiment."""

        id_rows = []
        nrows = 0
        ncols = 0
        for n in self.names_comps:
            c = n.split('|')
            c = [int(c[0]), int(c[1]), c[2]]
            if c[0] + 1 > nrows:
                nrows = c[0] + 1
                id_rows += [c[2]]
            if c[1] + 1 > ncols:
                ncols = c[1] + 1

        membs = np.array([1 * (i in comps_grp1) + 2 * (i in comps_grp2)
                          for i in range(self.n_comps)])
        cmap = ListedColormap(['w', 'b', 'r', 'g'])
        # cmap.set_bad("white")
        vmin = -0.5
        vmax = 3.5

        #print('self.names_comps :\n', self.names_comps)
        #print('membs:\n', membs)
        #print('nrows, ncols :', nrows, ncols)

        # see the dataloader classes for the convention in self.names_comps:
        mat = np.full([nrows, ncols], np.nan)
        for n, m in zip(self.names_comps, membs):
            c = n.split('|')[:2]
            i_row, i_col = int(c[0]), int(c[1])
            #print(n,m , i_row, i_col)
            mat[i_row, i_col] = m #if m > 0 else np.nan

        L = mat.astype('str')
        L[L!='0.0'] = '' 
        L[L=='0.0'] = 'x'
        sns.heatmap(
            mat, ax=ax, cmap=cmap, square=True, vmin=vmin, vmax=vmax,
            fmt='', annot=L,
            annot_kws={
                'fontsize': 4, 'color':'k', 'alpha': 1.0,
                'verticalalignment': 'center', 
                #'backgroundcolor': 'w'
                }
                )
        ax.set_yticks(np.arange(nrows) + 0.5)
        ax.set_yticklabels(id_rows, rotation=0)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 1.0, box.height * 0.9])

        return

    def viz_all(self, get_list):

        self.viz_agg1d()
        self.viz_single_grp(ind_comp=0)

        print(len(self.experiments))

        if get_list:
            '''
            we prefer not storing Traj2d instance, for memory reasons
            Since Traj2d.plot() invokes Traj2d.get_scores() which in turn invokes Traj2d._compute_quivers(),
            we first run self.viz_trajectories() here, and use its output to fill the scores part of self.experiments;
            hence running self.viz_scores() after self.viz_trajectories(). 

            '''
            print('\n------------ starting : .generate_list_experiments()')
            self.generate_list_experiments()
            self._report_len_experiments()
            print('------------ finished : .generate_list_experiments()')

            print('\n------------ starting : .compute_groups_traj2d()')
            self.compute_groups_traj2d()
            print('------------ finished : .compute_groups_traj2d()\n')

            #print('\n------------ starting : .compute_scores()')
            #self.compute_scores(num_grid_points = 80)
            #print('------------ finished : .compute_scores()\n')

        self.sort_experiments()
        
        self.viz_outputs_outliers()
        self.viz_outputs_comps()
        
        self.viz_trajectories()
        
        self.viz_scores()
        plt.show()

        with open(os.path.join(self.figs_path, 'scores.txt'), 'w') as fp:
            fp.write(",\n".join(
                ['Experiment %d : ' % i + '%f, %f' % getattr(exp_, 'scores') for i, exp_ in enumerate(self.experiments)]
            ))

        return

    def save(self):
        filename = os.path.join(self.runs_path, self.run_name + '.pt')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as file_:
            pickle.dump(self, file_, -1)
            print('saved as', filename)

    ###########

    def demo(self):
        if 1 == 1:
            '''
            Considering
                - the absolute values of "mags_outs"
                - folding them according to folding200 (taking cov over dim 1 and keeping dim 0, then averaging over dim 2)
                - sorting rows and columns based on the first singular vector
            we observe a two-planted clique pattern (this is in mags_outs).
            BUT the clustering is not very clear; we need clustering (not just sorting)
            '''

            data_ = np.abs(self.mags_outs)  # taking abs
            print(data_.shape)
            if data_.shape[1] > 100 and data_.shape[0] > 500:
                print('multiplication in a loop')
                mat = np.zeros((data_.shape[0], data_.shape[0]))
                for ii in range(data_.shape[2]):
                    if ii % 20 == 0:
                        print(ii)
                    mat += np.matmul(data_[:, :, ii], data_[:, :, ii].T)
            else:
                print('batch multiplication')
                mat = np.sum(np.matmul(
                    np.transpose(data_, (2, 0, 1)),
                    np.transpose(data_, (2, 1, 0))
                    ), axis=0)  # (n_comps, n_embd, n_embd) -> (n_embd, n_embd)
            
            svd = np.linalg.svd(mat, full_matrices=False)
            idx = np.argsort(np.abs(svd[0][:, 0]))
            fig, ax = plt.subplots(1, 1)
            sns.heatmap(mat[idx,:][:,idx], ax=ax, cmap=my_cmap, center=0)
            ax.set_title('sorting based on svd of folding200 of mags_outs')

        plt.show()

    ###########

    def get_total_score(
        self,
        num_grid_points,
        coords_partitioning,
        comps_grp1, comps_grp2=None
        ):
        '''
        TODO
        num_comps = C.traj_info_nd[mode]['num']
        _, _, scores, ind_outliers = create_2d_data(C, mode,
                       grp, [i for i in range(num_comps) if i not in grp],
                       to_id_outliers=True, to_score=True)
        return scores[0]-scores[1]
        '''
        if comps_grp2 is None:
            comps_grp2 = [i for i in range(self.n_comps) if i not in comps_grp1]
            
        scores = self.create_trajectories2d(
            comps_grp1, comps_grp2,
            coords_partitioning,
            method_agg=self.method_agg
            ).get_scores(num_grid_points)
        return scores[0] + scores[1]
    
    def find_components_grp(
        self,
        coords_partitioning,
        num_grid_points,
        max_iters):
        """Optimize the score returned by self.get_score over its set argument"""
        
        # print(get_total_score([]))

        grp = [0]
        
        scr = 100000
        log = []

        for iter_num in range(max_iters):
            print('\n\niter_num : ', iter_num)
            #c = [np.random.randint(self.n_comps, size=1).item()]
            sz_c = np.random.randint(self.n_comps // 50 + 2, size=1).item() + 3
            c = np.random.choice(list(range(self.n_comps)),
                                 size=sz_c,
                                 replace=False).tolist()
            c = sorted(list(set([j
                                 for i, a in enumerate(sorted(c))
                                 for j in range(a, a + 20)
                                 if j < self.n_comps])))
            if iter_num == 0:
                c = []
            print('random components:', c)

            '''
            if c in grp:
                scr_new = get_total_score(C, mode, [i for i in grp if i!=c])
            else:
                scr_new = get_total_score(C, mode, grp+[c])
            print('new score:', scr_new)

            if scr_new > scr:
                scr = scr_new
                if c in grp:
                    grp = [i for i in grp if i!=c]
                else:
                    grp = grp + [c]
            '''

            grp_new_options = [
                grp,
                list(set(grp + c)),
                list(set(grp + [0])),
                [i for i in c if i not in grp],
                [i for i in grp if i not in c],
                [i for i in grp if i in c],
                c,
                [i for i in list(set(grp+c)) if i not in [i for i in grp if i in c]]
                ]
            '''
            for j, grp_new in enumerate(grp_new_options):
                if len(grp_new) > 0 and len(grp_new)<self.n_comps:
                    scr_new = self.get_total_score(num_grid_points, coords_partitioning, grp_new)
                    if scr_new < scr:
                        scr = scr_new
                        grp = grp_new
            '''
            scr_ = np.zeros(len(grp_new_options)) + 100
            for j, grp_new in enumerate(grp_new_options):
                if len(grp_new) > 0 and len(grp_new) < self.n_comps:
                    scr_[j] = self.get_total_score(num_grid_points,
                                                   coords_partitioning,
                                                   grp_new
                                                   )
            i = np.argmin(scr_).tolist()
            scr = scr_[i]
            grp = grp_new_options[i]


            grp = [i for i in range(self.n_comps) if i in grp]
            grp_c = [i for i in range(self.n_comps) if i not in grp]
            if len(grp_c)<len(grp):
                grp = grp_c
            log = log + [(grp, scr)]
            print('>>>>>>>> iter_num, (grp, scr) : ', iter_num, log[-1])

            if iter_num%500 == 499:
                print('iter_num : ', iter_num)
                grp_c = [i for i in range(self.n_comps) if i not in grp]
                self.create_trajectories2d(
                    grp, grp_c,
                    coords_partitioning,
                    method_agg=self.method_agg
                    ).plot(
                        len_filter=None,
                        plot_quiver=True,
                        num_grid_points_quiver_plot=num_grid_points
                        )
                plt.show()

        for lg in log:
            print(lg)

        self.create_trajectories2d(
            comps_grp1, comps_grp2,
            coords_partitioning,
            method_agg=self.method_agg
            ).plot(
                len_filter=None,
                plot_quiver=True,
                num_grid_points_quiver_plot=num_grid_points
                )
        plt.show()

        return grp, scr
