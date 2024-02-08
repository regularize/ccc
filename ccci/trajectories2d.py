import numpy as np
import math

from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection

from copy import copy, deepcopy

from .utils import Quivs, id_outliers


class Trajectories2d():
    def __init__(
        self, 
        two_mats,
        coords_labels,
        coords_partitioning
        ):
        """

        Parameters:
        -----------
        two_mats : numpy array of size (n_embd, n_steps, 2)
        coords_labels : List
            a list of labels for coordinates, of length n_embd
        _score_frac_out : float
        _score_irr : List[Tuple[float]]
            a list of tuples of length 2 (score_irregularity, num_grid_points)
        _quivers : List[Quivs]
        """

        self.coords_labels = coords_labels

        self.n_embd = two_mats.shape[0]
        self.n_steps = two_mats.shape[1]
        assert two_mats.shape[2] == 2

        self.two_mats = two_mats

        # first ingredient for plotting and scores:
        self._score_frac_out = None
        if isinstance(coords_partitioning, str):
            self._find_outliers_traj2d(method=coords_partitioning)
        elif isinstance(coords_partitioning, dict):
            self._set_coords_in_out(
                coords_in=coords_partitioning['coords_in'],
                coords_out=coords_partitioning['coords_out']
                )

        # second ingredient for plotting and scores:
        self._quivers = []

    def _get_id_quivers(self, num_grid_points):
        i_q = [
            i for i, quiv_info in enumerate(self._quivers)
            if getattr(quiv_info, 'num_grid_points') == num_grid_points
            ]
        if i_q == []:
            self._compute_quivers(num_grid_points)
            i_q = -1
        else:
            i_q = i_q[0]
            print('quiver information exists.')
        return i_q

    ##########

    def _set_coords_in_out(self, coords_in, coords_out):
        self._coords_in = coords_in
        self._coords_out = coords_out
        self._score_frac_out = len(self._coords_out) / self.n_embd

    def _find_outliers_traj2d(self, method):
        """Find outliers, only using information available
        in a 2d trajectory plot of aggregated magnitudes.

        Note that it does not matter if we first center a cloud of points
        before projecting them onto a line and taking MAD,
        as MAD will not care about a shift.
        """

        print('running _find_outliers_traj2d(); for a given 2d trajectories data, based on method_find_outliers_traj2d of', method)
        output = None

        if method == 'random':
            frac = 0.05
            print('  find_outliers_traj2d() randomly; a fraction of ', frac)
            coords_out = np.random.randint(
                0, self.n_embd,
                (math.ceil(self.n_embd * frac),)
                ).tolist()

        elif method == 'MAD-1median-2dif':
            print('  find_outliers_traj2d() based on: id_outliers(median(mags1)-median(mags2)), at the last time step')
            output = id_outliers(
                np.median(self.two_mats[:, :, 0], axis=1)
                - np.median(self.two_mats[:, :, 1], axis=1)
                )

        elif method == 'MAD-2median-1dif':
            print('  find_outliers_traj2d() based on: id_outliers(median(mags1-mags2)), at the last time step')
            output = id_outliers(
                np.median(
                    self.two_mats[:, :, 0]
                    - self.two_mats[:, :, 1],
                    axis=1)
                )

        elif method == 'MAD-last-dif':
            print('  find_outliers_traj2d() based on: id_outliers(mags1-mags2), at the last time step')
            output = id_outliers(
                self.two_mats[:, -1, 0]
                - self.two_mats[:, -1, 1]
                )

        elif method == 'MAD-last-sum':
            print('  find_outliers_traj2d() based on the 1-d aggregation of all components, at the last time step')
            output = id_outliers(np.sum(self.two_mats[:, -1, :], axis=-1))

        elif method == 'MAD-every30-lincomb':
            num_lincomb = 150
            print('  find_outliers_traj2d() based on the union over every 30 time steps (via id_outliers({0} linear combinations of 2d agg_mags))'.format(num_lincomb))
            coords_out_union = []
            for j in range(0, self.two_mats.shape[1], 30):
                for r in range(num_lincomb):
                    a = np.random.rand(1) * 2 * math.pi
                    output = id_outliers(
                        self.two_mats[:, j, 0] * np.cos(a).item()
                        + self.two_mats[:, j, 1] * np.sin(a).item()
                        )
                    coords_out_union = coords_out_union + output['coords_out']
            coords_out = [i for i in range(self.n_embd)
                          if i in coords_out_union]

        elif method == 'MAD-union-lincomb':
            num_lincomb = 20
            print('  find_outliers_traj2d() based on the union over all time steps (via id_outliers({0} linear combinations of 2d agg_mags))'.format(num_lincomb))
            coords_out_union = []
            for j in range(self.two_mats.shape[1]):
                for r in range(num_lincomb):
                    a = np.random.rand(1) * 2 * math.pi
                    output = id_outliers(
                        self.two_mats[:, j, 0] * np.cos(a).item()
                        + self.two_mats[:, j, 1] * np.sin(a).item()
                        )
                    coords_out_union = coords_out_union + output['coords_out']
            coords_out = [i for i in range(self.n_embd)
                          if i in coords_out_union]

        elif method == 'MAD-union-dif':
            print('  find_outliers_traj2d() based on the union over all time steps (via id_outliers(mags1-mags2))')
            coords_out_union = []
            for j in range(self.two_mats.shape[1]):
                output = id_outliers(
                    self.two_mats[:, j, 0]
                    - self.two_mats[:, j, 1]
                    )
                coords_out_union = coords_out_union + output['coords_out']
            coords_out = [i for i in range(self.n_embd)
                          if i in coords_out_union]

        elif method == 'MAD-union-sum':
            print('  find_outliers_traj2d() based on the union over all time steps (via id_outliers(mags1+mags2))')
            coords_out_union = []
            for j in range(self.two_mats.shape[1]):
                output = id_outliers(
                    self.two_mats[:, j, 0]
                    + self.two_mats[:, j, 1]
                    )
                coords_out_union = coords_out_union + output['coords_out']
            coords_out = [i for i in range(self.n_embd)
                          if i in coords_out_union]

        else:
            raise NotImplementedError

        # unifying the results of different methods:
        if output is None and coords_out:
            coords_in = [i for i in range(self.n_embd)
                         if i not in coords_out]
            output = {'coords_in': coords_in,
                      'coords_out': coords_out}
        if output is None:
            raise TypeError

        self._set_coords_in_out(output['coords_in'], output['coords_out'])

    def get_coords_out(self):
        return self._coords_out

    ##########

    def get_scores(self, num_grid_points, verbose=True):
        print('running get_scores')
        i_q = self._get_id_quivers(num_grid_points)
        _score_irr = getattr(self._quivers[i_q], 'irr_score')
        if verbose:
            print('   irregularity(num_grid_points={0:3d}) : {1:1.5f}'.format(num_grid_points, _score_irr))
            print('   fraction of outliers            : {0:1.5f}'.format(self._score_frac_out))
        return [_score_irr, self._score_frac_out]  # recall that outliers are set in __init__

    def plot(self, len_filter, plot_quiver, num_grid_points_quiver_plot):

        print('-- running plot() of class Trajectories2d')
        scores = (None, None)

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs = axs.flatten()

        fig.subplots_adjust(wspace=0, hspace=0)
        # plt.setp(axs, xticks=[], yticks=[])

        x = self.two_mats[:, :, 0].T
        y = self.two_mats[:, :, 1].T
        if self._coords_out is None:
            print('coords_in and coords_out have not been specified.')
            raise AttributeError

        if 1 == 0:
            # hist2d (before any smoothing of the trajectories):
            axs[0].hist2d(x.flatten(), y.flatten(), bins=100, cmap='binary')
            sns.kdeplot(
                {'vals': x.flatten(), 'mags': y.flatten()},
                x='vals', y='mags', fill=True, ax=ax
                )

        if len_filter is not None:
            # smoothing the trajectories:
            def filter_(m):
                return np.convolve(m, np.ones(len_filter) / len_filter, mode='valid')
            x = np.apply_along_axis(filter_, axis=0, arr=x)
            y = np.apply_along_axis(filter_, axis=0, arr=y)

        # axs[0]: inliers
        cmap = plt.get_cmap('tab20')
        for i_lab, lab in enumerate(set(self.coords_labels)):
            coords_in_lab = [c for i, c in enumerate(self._coords_in)
                             if self.coords_labels[i] == lab]
            axs[0].plot(
                x[:, coords_in_lab], y[:, coords_in_lab],
                '-', color=cmap(i_lab / len(set(self.coords_labels))),
                alpha=0.5, linewidth=1.0, markersize=0.5
                )        
        '''
        axs[0].plot(
            np.mean(x[:, self._coords_in], axis=-1),
            np.mean(y[:, self._coords_in], axis=-1),
            'k-', alpha=1.0, linewidth=2, markersize=0.5
            )
        '''
        ## to show a gradient over the average line corresponding to checkpoint
        xx = np.mean(x[:, self._coords_in], axis=-1).flatten()
        yy = np.mean(y[:, self._coords_in], axis=-1).flatten()
        points = np.array([xx, yy]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments, 
            array=np.arange(len(xx)),
            linestyle='-', alpha=1.0, linewidth=4, 
            cmap='Reds',
            )
        lc_copy = deepcopy(lc) # making a copy before being added to an ax
        axs[0].add_collection(lc)


        # axs[1]: outliers
        for i_lab, lab in enumerate(set(self.coords_labels)):
            coords_out_lab = [c for i, c in enumerate(self._coords_out)
                              if self.coords_labels[i] == lab]
            axs[1].plot(
                x[:, coords_out_lab], y[:, coords_out_lab],
                '-', color=cmap(i_lab / len(set(self.coords_labels))),
                alpha=0.5, linewidth=1.0, markersize=0.5
                )
        rect = patches.Rectangle(
            (axs[0].get_xlim()[0], axs[0].get_ylim()[0]), 
            axs[0].get_xlim()[1] - axs[0].get_xlim()[0], 
            axs[0].get_ylim()[1] - axs[0].get_ylim()[0], 
            fill=False, linewidth=2, edgecolor='k', facecolor='none'
            )
        axs[1].add_patch(rect)
        axs[1].add_collection(lc_copy)

        # axs[2]: quiver
        if plot_quiver:

            scores = self.get_scores(num_grid_points_quiver_plot, verbose=False)
            axs[2].set_title(
                'scores (with {0:d} grid points): ({1:.3f},{2:.3f})'.format(
                    num_grid_points_quiver_plot, *scores
                    ),
                size=8
                )

            # quiver information might have been computed outside of .plot() here,
            # or in .get_score() above when producing plot title.
            i_q = self._get_id_quivers(num_grid_points_quiver_plot)
            pnts, Quivs, Cols = self._quivers[i_q].get_fields([
                'points', 'ave_quivers', 'density'
                ])
            axs[2].quiver(
                pnts[0, :, :], pnts[1, :, :], Quivs[0, :, :], Quivs[1, :, :],
                np.power(np.divide(Cols, np.max(Cols) + 1e-6), 0.2),
                angles='xy',
                units='width',
                # scale_units='xy',
                scale=num_grid_points_quiver_plot/100,
                pivot='tail', cmap='binary',
                linewidth=4*Cols.flatten(),
                edgecolors='k',
                )

        return scores, fig, axs

    ##########

    def _compute_quivers(self, num_grid_points):
        """Compute average quivers, for the displacement vectors
        from all inlier trajectories.
        Also compute an irregularity score for the trajectories.
        """

        def lens_rows(A):
            return np.sqrt(np.power(A, 2).sum(axis=1))

        def lens(A):
            return np.sqrt(np.power(A, 2).sum(axis=0))

        print('  running _compute_quivers for inlier trajectories.')
        x = self.two_mats[self._coords_in, :, 0].T
        y = self.two_mats[self._coords_in, :, 1].T

        # inputs:
        k1 = 2
        k2 = x.shape[0] - 1
        D4 = np.concatenate(
            (x[k1 - 1: k2, :].flatten()[:, None],
             y[k1 - 1: k2, :].flatten()[:, None],
             (x[k1: k2 + 1, :] - x[k1 - 1: k2, :]).flatten()[:, None],
             (y[k1: k2 + 1, :] - y[k1 - 1: k2, :]).flatten()[:, None] ),
            axis=1
            )

        # outputs:
        bins = [num_grid_points, num_grid_points]

        ret0 = stats.binned_statistic_2d(
            D4[:, 0], D4[:, 1], D4[:, 2], 'mean', bins)
        ret1 = stats.binned_statistic_2d(
            D4[:, 0], D4[:, 1], D4[:, 3], 'mean', bins)
        Q = np.concatenate(
            (np.nan_to_num(ret0.statistic)[None, :, :],
             np.nan_to_num(ret1.statistic)[None, :, :]),
            axis=0
            )

        ret = stats.binned_statistic_2d(
            D4[:, 0], D4[:, 1], None, 'count', bins
            )
        C = ret.statistic
        C = C / (np.sum(C) + 1e-8)

        ret_lens = stats.binned_statistic_2d(
            D4[:, 0], D4[:, 1], lens_rows(D4[:, 2:]), 'mean', bins
            )
        L = np.nan_to_num(ret_lens.statistic)

        grid_x, grid_y = np.meshgrid(
            0.5 * (ret.x_edge[:-1] + ret.x_edge[1:]),
            0.5 * (ret.y_edge[:-1] + ret.y_edge[1:])
            )
        P = np.concatenate(
            (grid_x.T[None, :, :],
             grid_y.T[None, :, :]),
            axis=0
            )

        # a score between 0 and 1:
        sc = 1.0 - np.sum(np.multiply(C, np.divide(lens(Q), L + 1e-6)))
        # so that in case there exists an averaged out area, then we get a high score
        # if using mean, then existence of some averaged out areas can still result in a low score

        self._quivers += [Quivs(num_grid_points, P, Q, C, L, sc)]
        return
