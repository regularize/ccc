import numpy as np

from .utils import MAD_scores_1sided
from .trajectories2d import Trajectories2d


class CCCBase():
    """Store 
        Coordinate values,
        over multiple Checkpoint,
        across different Components,
    of a neural network, along with some metadata, 
    and provide 
        relevant summarization methods for this 3-d array. 
    """

    def __init__(
        self,
        model_name,
        steps_list,
        traj_info_nd
        ):
        self.model_name = model_name

        mags_set = traj_info_nd['mags_set']
        assert len(mags_set) == len(steps_list)
        assert all(isinstance(b, bool) for b in mags_set)

        self.steps_list = [s for i, s in enumerate(steps_list) if mags_set[i]]
        self.mags = np.array(traj_info_nd['mags'][:, mags_set, :])

        self.n_embd = self.mags.shape[0]
        self.n_steps = self.mags.shape[1]
        self.n_comps = self.mags.shape[2]

        self.names_comps = traj_info_nd['names']
        self.weights_comps = np.array(traj_info_nd['lengths'])

    def _dim1_reduce(self, attr, method):
        if method == 'median':
            return np.median(getattr(self, attr), axis=1)
        elif method == 'last':
            return getattr(self, attr)[:, -1, :]
        else:
            raise NotImplementedError
        return

    def _dim2_aggregate_to_2(
        self, attr, 
        comps_grp1, comps_grp2, 
        method_agg
        ):
        data = getattr(self, attr) # useful for when subclasses store new attributes
        two_mats = np.zeros((data.shape[0], data.shape[1], 2))
        w1 = self.weights_comps[comps_grp1]
        w2 = self.weights_comps[comps_grp2]
        if method_agg == 'lengths-weighted':
            w1 = w1 / np.sum(w1)
            w2 = w2 / np.sum(w2)
            two_mats[:, :, 0] = data[:, :, comps_grp1].sum(axis=-1)
            two_mats[:, :, 1] = data[:, :, comps_grp2].sum(axis=-1)
        elif method_agg == 'unweighted':
            w1 = w1 / np.sum(w1)
            w2 = w2 / np.sum(w2)
            two_mats[:, :, 0] = np.divide(
                data[:, :, comps_grp1],
                w1
                ).sum(axis=-1)
            two_mats[:, :, 1] = np.divide(
                data[:, :, comps_grp2],
                w2
                ).sum(axis=-1)
        else:
            raise NotImplementedError
        return two_mats


class CCC(CCCBase):
    """Preprocess and 
    prepare-for-visualization (by instantiating Trajectories2d) 
    the CCC-data (a 3-d array).
    """
    def __init__(
        self,
        model_name,
        steps_list,
        traj_info_nd,
        **kwargs
        ):
        """
        Parameters:
        -----------
            model_name: str
            steps_list: List[int]
            traj_info_nd: dict
            kwargs: optional
                for things like n_layer, n_head, etc; e.g., for visualization

            mags_n
            mags_outs
        """
        super(CCC, self).__init__(model_name,
                                                         steps_list,
                                                         traj_info_nd)
        self.kwargs = kwargs
        self.set_mags_outs()

    def set_mags_outs(self):
        """Prepare raw self.mags for TriClassification
        by applying certain transformations.
        Set the attribute self.mags_outs.
        """

        def normalize_by_max(p):
            return np.divide(p, np.max(p, axis=0, keepdims=True) + 1e-4)

        # we treat each aggregated component as one piece of information,
        # hence ignoring self.weights_comps:
        self.mags_n = normalize_by_max(self.mags)

        self.mags_outs = MAD_scores_1sided(self.mags_n, dm=0)
        self.mags_outs = np.clip(self.mags_outs, -6.0, 6.0)
        # to enhance the intensities:
        self.mags_outs = np.sign(self.mags_outs) * np.power(self.mags_outs, 4)
        # normalize across coordinates by inf norm:
        self.mags_outs = (np.sign(self.mags_outs)
                          * normalize_by_max(np.abs(self.mags_outs)))

        return

    def create_trajectories2d(self,
                              comps_grp1, comps_grp2,
                              coords_partitioning,
                              method_agg):
        """Aggregate the 3d array self.mags (the CCC-data) based on the two component groups,
        and create an instance of the class Trajectories2d and return it.
        Pass coords_partitioning to this instance of Trajectories2d.

        Parameters:
        -----------
            comps_grp1: List[int]
                a subset of components
            comps_grp2: List[int]
                a subset of components
            coords_partitioning: str | dict
                either a string, providinf the outlier detection method
                    to be passed to an instance of Trajectories2s
                or a dictionary with keys 'coords_in' and 'coords_out'
        Returns:
        --------
            Traj2d: Trajectories2d
        """

        # config: 
        traj2d_normalize = True
        traj2d_log = True
        traj2d_log = traj2d_log and traj2d_normalize
        print(
            ('running create_trajectories2d() '
            + 'with aggregation method {0:s}, '
            + 'traj2d_normalize {1}, traj2d_log {2}').format(
                method_agg, traj2d_normalize, traj2d_log
                )
                )

        assert (isinstance(coords_partitioning, str)
                or (isinstance(coords_partitioning, dict)
                    and 'coords_in' in coords_partitioning
                    and 'coords_out' in coords_partitioning))

        two_mats = self._dim2_aggregate_to_2(
            'mags',
            comps_grp1, comps_grp2,
            method_agg=method_agg
            )
        assert two_mats.shape == (self.mags.shape[0], self.mags.shape[1], 2)

        if traj2d_normalize:
            two_mats = two_mats - np.min(two_mats, axis=(0, 1))[None, None, :]
            two_mats = np.divide(
                two_mats,
                np.max(two_mats, axis=(0, 1))[None, None, :] + 1e-6
                )

            if traj2d_log:
                two_mats = np.log(two_mats + 0.01)

        coords_labels = [c // self.kwargs['n_head']
                         for c in range(self.n_embd)]
        Traj2d = Trajectories2d(two_mats, coords_labels, coords_partitioning)

        return Traj2d
