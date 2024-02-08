import numpy as np

from itertools import groupby, chain, combinations
from operator import itemgetter
from typing import Tuple
from dataclasses import dataclass, fields

import seaborn as sns
my_cmap = sns.diverging_palette(
    h_neg=150,
    h_pos=10,
    s=90,
    l=50,
    sep=1,
    as_cmap=True
    )


def class_str(class_name, list_str_pairs):
    """A convenient way to print several key-value pairs for an instance of a class."""
    max_key_length = max([len(k) for k,v in list_str_pairs])
    return (
        class_name + 
        '(' +
        (',\n' + ' ' * (len(class_name) + 1)).join(
            [('{0:'+str(max_key_length)+'s}: {1!r}').format(k, v)
            for k, v in list_str_pairs]
            ) + 
            ')'
            )

class Groups:
    """A dictionary and some access and display methods."""
    def __init__(self, groups):
        self.__attr_list = ['coords_in', 'coords_out',
                            'comps_grp1', 'comps_grp2']
        if (isinstance(groups, dict)
            and set(groups.keys()) <= set(self.__attr_list)
            and all([isinstance(v, list) for k, v in groups.items()])):
            self.groups = groups
        else:
            raise TypeError(
                'class Groups can only be instantiated with'
                + 'a dictionary with keys from '
                + ', '.join(self.__attr_list)
                + ', and values that are lists.'
                )

    def has_all_keys(self):
        return set(self.groups.keys()) == set(self.__attr_list)

    def has_coords_keys(self):
        return set(self.groups.keys()) >= set(['coords_in', 'coords_out'])

    def has_comps_keys(self):
        return set(self.groups.keys()) >= set(['comps_grp1', 'comps_grp2'])

    def __repr__(self):
        """Provide a useful summary."""
        return ('Groups('
                + ', '.join(
                    ['len({0:s})={1:d}'.format(k, len(v))
                     for k, v in self.groups.items()]
                    ) + ')'
                )

    def __str__(self):
        """Provide a pretty print functionality."""
        str_outs = []
        for attr in self.__attr_list:
            lst = self.groups.get(attr, None)
            if lst is not None:
                str_outs += ['  {0:10s} (length: {1:3d}): {2}'.format(
                    attr, len(set(lst)), self.__list_as_intervals(lst))]
        if str_outs == []:
            return('  an empty instance of Groups')
        return '\n'.join(str_outs)

    def __list_as_intervals(self, lst):
        ranges = []
        for k, g in groupby(
            enumerate(sorted(set(lst))),
            lambda x: x[0] - x[1]
            ):
            group = (map(itemgetter(1), g))
            group = list(map(int, group))
            new_range = str(group[0])
            if len(group) > 1:
                new_range += '-' + str(group[-1])
            ranges += [new_range]
        return ', '.join(ranges)


@dataclass(frozen=True)
class Experiment:
    source: str = ''
    method: str = ''
    detail: str = ''
    groups: Groups = None
    scores: Tuple[float, float] = (None, None)

    def __str__(self):
        return class_str(
            self.__class__.__name__, 
            [(field.name, getattr(self, field.name)) for field in fields(Experiment)]
            )

@dataclass(frozen=True)
class Quivs:
    """
    Parameters:
    -----------
    num_grid_points:  int
        number of grid points in either directions.
    points:         np.ndarray of size (num_grid_points, num_grid_points)
        points on a grid
    ave_quivers:    np.ndarray of size (2, num_grid_points, num_grid_points)
        average displacement vectors as quivers on the grid P
    density:        np.ndarray of size (num_grid_points, num_grid_points)
        normalized counts of displacement vectors with origins in each point in P
    ave_lengths:    np.ndarray of size (num_grid_points, num_grid_points)
        average length of displacement vectors on the grid P
    irr_score: float
        irregularity score for the quivers
    """
    num_grid_points: int
    points: np.ndarray = None
    ave_quivers: np.ndarray = None
    density: np.ndarray = None
    ave_lengths: np.ndarray = None
    irr_score: float = None

    def get_fields(self, field_names):
        assert set(list(field_names)) <= set([field.name for field in fields(Quivs)])
        return tuple(getattr(self, fn) for fn in field_names)
    
    def __str__(self):
        return (
            self.__class__.__name__ 
            + '('
            + 'num_grid_points={0:d}, '.format(getattr(self, 'num_grid_points'))
            + (
                'quiver information does not exist, '
                if any(
                    [
                        getattr(self, field.name) is None
                        for field in fields(Quivs)
                        if field.name!='irr_score'
                        ]
                        )
                        else 'quiver information exists, '
                        )
                        + (
                            'irr_score={0:1.4f}'.format(sc)
                            if (sc := getattr(self, 'irr_score')) is not None
                            else 'irr_score=None'
                            )
                            + ')'
                            )


def cluster_1d(z):
    assert z.ndim == 1

    z_srt_values = np.sort(z, kind='stable')
    z_srt_indices = np.argsort(z, kind='stable')

    costs = np.zeros(z.shape)
    for i in range(*z.shape):
        # i is the number of elements in the first bracket
        if i > 0:
            costs[i] += np.sum(np.power(z_srt_values[:i]
                                        - np.mean(z_srt_values[:i]), 2))
        costs[i] += np.sum(np.power(z_srt_values[i:]
                                    - np.mean(z_srt_values[i:]), 2))
    i_brk = np.argmin(costs)

    grp1 = z_srt_indices[:i_brk].tolist()
    grp2 = z_srt_indices[i_brk:].tolist()

    return grp1, grp2


def MAD_scores_1sided(v, dm):
    dev = v - np.median(v, axis=dm, keepdims=True)
    mad = np.median(np.abs(dev), axis=dm, keepdims=True)  # Median Absolute Deviation
    return np.divide(dev, 1e-6 + mad / 0.6745)


def MAD_scores(v, dm):
    dev = v - np.median(v, axis=dm, keepdims=True)
    mad = np.median(np.abs(dev), axis=dm, keepdims=True)  # Median Absolute Deviation
    return np.divide(np.abs(dev), mad / 0.6745 + 1e-6)


def MAD_outlier_det(v, dm, thr):
    """Identify outlier entries, judged along dimension dm.
    
    Paramerers:
    -----------
        v: numpy array
        dm: one of v's dimensions
        thr: threshold for MAD scores

    Returns:
    --------
        a boolean array of the same size as v.
    """
    assert 0 <= dm and dm < v.ndim
    return MAD_scores(v, dm=dm) > thr


def id_outliers(v):
    """Identify indices of outlier entries.

    Parameters:
    -----------
        v: a 1-dimensional numpy array

    Returns:
    --------
        a dictionary with keys 'coords_in' and 'coords_out'
    """
    assert v.ndim == 1
    thr = 3.5
    coords_out = np.nonzero(MAD_outlier_det(v, dm=0, thr=thr))[0].tolist()
    coords_in = [i for i in range(v.shape[0]) if i not in coords_out]
    return {'coords_in': coords_in, 'coords_out': coords_out}
