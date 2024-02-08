import numpy as np
import math

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralBiclustering, SpectralCoclustering, AgglomerativeClustering

from functools import wraps

from .utils import MAD_outlier_det, id_outliers, cluster_1d, my_cmap
from .utils import Groups, Experiment

## some of the 'folding' methods are inspired by the paper "Tensor biclustering" by Feizi et al, 2017, NeurIPS. 


class TriClassification():
    """Provide various methods to cluster 3-dimensional data.

    For clarity, we name the 3 dimensions as follows:
        0: coordinates
        1: checkpoints
        2: components

    In the code:
        (coords_in, coords_out)  refer to a partition of dim 0
            (disjoint and complementary).
        (comps_grp1, comps_grp2) refer to two subsets of dim 2
            (not necessarily disjoint or complementary).

    Methods that start with '_inp',
        take two arguments 'data' and 'method',
        return two outputs 'data' and 'output'
    where 'output' is a dictionary with keys from
        ['coords_in', 'coords_out', 'comps_grp1', 'comps_grp2'].

    Notes:
    ------
    Several methods start with '_inp', 
        accept two arguments 'data' and 'method',
        and return 'data' and 'output'.

    Some produce an 'output' and return None for 'data';
        e.g., _inp101_out100;
    Some modify the data and return None forr 'output';
        e.g., _inp111_data101.

    Here is the naming convention:
        As an example, consider self._inp101_out100(data, method):
        '101' in 'inp101' indicates that the input 'data' has two dimensions,
        and its dimensions correspond to first and third dimensions
        of the original 3d tensor in self.data (coordinates and components).
        '100' in 'out100' indicates that the 'output' will contain groupings for
        the first dimension of the original 3d tensor (coordinates).

        As an example, consider self._inp111_data101(data, method):
        '111' in 'inp111' indicates that the input 'data' has three dimensions.
        '101' in 'data101' indicates that the 'output' will contain groupings for
        the first and third dimensions of the original 3d tensor, namely
        all four possible keys will be present in the dictionary.
    """

    def __init__(self, data, weights):
        assert data.ndim == 3
        assert weights.ndim == 1
        assert weights.shape[0] == data.shape[2]
        self.data = data
        self.weights = weights

    ##########

    def _print_methodarg(self, function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            print(
                ' ' * 4
                + '-- calling {0:16s}'.format(function.__name__)
                + ' with data shape', args[0].shape,
                'and input method:', args[1]
                )
            try:
                assert not np.isnan(args[0]).any()
            except Exception as e:
                print('ERROR: The first argument contains NaN values.')
            result = function(*args, **kwargs)
            return result
        return wrapper

    def __getattribute__(self, attr):
        method = object.__getattribute__(self, attr)
        if callable(method) and attr.startswith('_inp'):
            # print(' '*4 + '---- calling:', attr)#, ' | method :', method)
            method = self._print_methodarg(method)
        return method

    def classify(self, instructions):
        """Perform classification according to the given instructions.

        Several methods of TriClassification, starting with "_inp", take as arguments:
            data, method,
        and return
            data, output,
        where output is a dictionary with keys from
            ['coords_in', 'coords_out', 'comps_grp1', 'comps_grp2'].
        
        We loop through a list of such 'instructions'.

        Some class methods might return "None, output".

        Parameters:
        -----------
        instructions: List[Tuple]
            tuple[0]: name of class method of TriClassification
            tuple[1]: the "method" argument that the class method takes in

        Returns:
        --------
        output: dict
            A dictionary with keys from
            ['coords_in', 'coords_out', 'comps_grp1', 'comps_grp2']
        """
        print('executing the following instructions:', instructions)
        data = self.data  # 3-dimensional
        output = {}
        for inst in instructions:
            if not inst[0].startswith('_inp'):
                print('Not a valid instruction for classify().')
                raise AttributeError
            func = getattr(self, inst[0])
            data, output = func(data, inst[1])
        return output

    def _instr_to_str(self, instr):
        """Produce a string for an instructions list."""
        assert isinstance(instr, list)
        assert all([
            isinstance(p, tuple)
            and len(p) == 2
            and isinstance(p[0], str)
            and isinstance(p[1], str) for p in instr]
            )
        return (
            'instructions[' + ', '.join(
                ['({0:s}, {1:s})'.format(p[0], p[1]) for p in instr]
                )
                + ']'
                )

    def _coords_in_from_out(self, coords_out):
        """Produce inlier coordinates list given outlier coordinates list."""
        assert set(coords_out) <= set(range(self.data.shape[0]))
        return [i for i in range(self.data.shape[0]) if i not in coords_out]

    def _comps_complement(self, comps_grp):
        """Produce the complement of a components list."""
        assert set(comps_grp) <= set(range(self.data.shape[2]))
        return [i for i in range(self.data.shape[2]) if i not in comps_grp]
    
    ##########

    def _inp111_data101(self, data, method):
        if method == 'median':
            return np.median(data, axis=1), None
        elif method == 'last':
            return data[:, -1, :], None
        else:
            raise NotImplementedError('requested method has not been implemented in _inp111_data101().')
        return

    def _inp101_data100(self, data, method):
        if method == 'rows_L2sq':
            return np.power(data, 2).sum(axis=-1), None
        elif method == 'rows_L2':
            return np.sqrt(np.power(data, 2).sum(axis=-1)), None
        elif method == 'rows_L1':
            return np.abs(data).sum(axis=-1), None
        else:
            raise NotImplementedError('requested method has not been implemented in _inp101_data100().')
        return

    def _inp101_data101(self, data, method):
        if method == 'normalize_rows_byL1':
            return np.divide(
                data, 
                np.sum(data, axis=-1, keepdims=True) + 1e-8
                ), None
        elif method == 'normalize_cols_byL2':
            return np.divide(
                data,
                np.sqrt(np.sum(np.power(data, 2), axis=0, keepdims=True)) + 1e-8
                ), None
        else:
            raise NotImplementedError('requested method has not been implemented in _inp101_data101().')
        return

    def _my_matmul(self, mat1, mat2):
        """Batch matrix multiplication; 
        utilizing GPU if available; 
        using a loop over the batch dimension if needed.
        """
        n_ave, n_keep, n_rem = mat1.shape
        assert (n_ave, n_rem, n_keep) == mat2.shape

        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():

            print('  via multiplication in a loop with GPU')
            mat = torch.zeros(n_keep, n_keep).to(device)
            for ii in range(n_ave):
                if ii % 100 == 0:
                    print(' ', ii)
                mat += torch.matmul(
                    torch.tensor(mat1[ii, :, :]).to(device),
                    torch.tensor(mat2[ii, :, :]).to(device)
                    )
            mat = mat.detach().cpu().numpy()

        elif n_ave < 100 or n_keep < 500:

            print('  via batch multiplication')
            mat = np.sum(np.matmul(mat1, mat2), axis=0) 
            # (n_d_ave, n_d_keep, n_d_keep) -> (n_d_keep, n_d_keep)
        
        else:
        
            print('  via multiplication in a loop')
            mat = np.zeros((n_keep, n_keep))
            for ii in range(n_ave):
                if ii % 100 == 0:
                    print(' ', ii)
                mat += np.matmul(mat1[ii, :, :], mat2[ii, :, :])

        return mat

    def _folding(self, data, method):
        assert not np.isnan(data).any()

        assert (
            isinstance(method, str) and 
            len(method) == 3 and 
            method.isdigit()
            )
        d_ave = int(method[0])
        d_keep = int(method[1])
        d_rem = int(method[2])
        assert set([d_ave, d_keep, d_rem]) == set([0, 1, 2])

        try:
            mat = getattr(self, 'folding' + method)
            print(' folding' + method + ': retrieved')
        except Exception as e:
            print(e)

            print(data.shape)
            print(' folding' + method + ':')
            mat = self._my_matmul(
                np.transpose(data, (d_ave, d_keep, d_rem)), 
                np.transpose(data, (d_ave, d_rem, d_keep))
            )

            setattr(self, 'folding' + method, mat)

        assert not np.isnan(mat).any()
        assert np.allclose(mat, mat.T, rtol=1e-05, atol=1e-08)
        

        return mat

    ########## method that take in a "vector", output (inliers, outliers)
    
    def _inp100_out100(self, data, method):
        assert data.ndim == 1
        if method == 'MAD':
            return None, id_outliers(data)
        else:
            raise NotImplementedError('requested method has not been implemented in _inp100_out100().')
        return

    ########## method that take in a "matrix", output (inliers, outliers) and (cols_grp1,cols_grp2)
    
    def _inp101_out101(self, mat, method):
        assert mat.ndim == 2
        comps_grp1 = None

        if method == 'bipartition_Dh_normalizeTrue':
            ## "Co-clustering documents and words using bipartite spectral graph partitioning, Dhillon"
            mat = mat - np.min(mat)  #### might be a bad thing to do, but Dhillon's algorithm needs it
            ##### we might be able to use Signed Community Detection algorithms....  TODO
            d1 = np.sum(mat, axis=1)
            d2 = np.sum(mat, axis=0)
        if method == 'bipartition_Dh_normalizeFalse':
            ## "Co-clustering documents and words using bipartite spectral graph partitioning, Dhillon"
            mat = mat - np.min(mat)  #### might be a bad thing to do, but Dhillon's algorithm needs it
            ##### we might be able to use Signed Community Detection algorithms....  TODO
            d1 = np.ones_like(np.sum(mat, axis=1))
            d2 = np.ones_like(np.sum(mat, axis=0))
            print('     created d1,d2')

        if method.startswith('bipartition_Dh'):
            d1_mh = np.divide(1.0, np.sqrt(d1))
            d2_mh = np.divide(1.0, np.sqrt(d2))
            mat_n = d1_mh[:, None] * mat * d2_mh[None, :]
            svd = np.linalg.svd(mat_n, full_matrices=False)
            z = np.hstack((svd[0][:, 1] * d1_mh, svd[2][1, :] * d2_mh))

            grp1, grp2 = cluster_1d(z)

            rows_grp1 = [i for i in range(d1.shape[0]) if i in grp1]
            rows_grp2 = [i for i in range(d1.shape[0]) if i in grp2]
            if len(rows_grp1) >= len(rows_grp2):
                coords_in = rows_grp1
                coords_out = rows_grp2
            else:
                coords_in = rows_grp2
                coords_out = rows_grp1
            comps_grp1 = [i for i in range(d2.shape[0]) if i + d1.shape[0] in grp1]
            comps_grp2 = [i for i in range(d2.shape[0]) if i + d1.shape[0] in grp2]

            output = {'coords_in': coords_in, 'coords_out': coords_out,
                      'comps_grp1': comps_grp1, 'comps_grp2': comps_grp2}

        if method == 'bipartition_spectral':
            model = SpectralBiclustering(
                n_clusters=(2, 2),
                method="log",
                random_state=0
                )
            model.fit(mat)

            rows_grp1 = (model.row_labels_ == 0).nonzero()[0].tolist()
            rows_grp2 = (model.row_labels_ == 1).nonzero()[0].tolist()
            if len(rows_grp1) >= len(rows_grp2):
                coords_in = rows_grp1
                coords_out = rows_grp2
            else:
                coords_in = rows_grp2
                coords_out = rows_grp1
            comps_grp1 = (model.column_labels_ == 0).nonzero()[0].tolist()
            comps_grp2 = (model.column_labels_ == 1).nonzero()[0].tolist()

            output = {'coords_in': coords_in, 'coords_out': coords_out,
                      'comps_grp1': comps_grp1, 'comps_grp2': comps_grp2}

        if method == 'coords:L1-MAD/comps:svd-sign':
            _, output_coords = self._inp100_out100(
                self._inp101_data100(mat, 'rows_L1')[0],
                'MAD'
                )
            _, output_comps = self._inp101_out001(mat, 'svd-sign')
            output = dict(output_coords, **output_comps)

        if not ('coords_in' in output and 'coords_out' in output
                and 'comps_grp1' in output and 'comps_grp2' in output):
            raise NotImplementedError('requested method has not been implemented in _inp101_out101().')

        return None, output

    ########## method that take in a "matrix", output (inliers, outliers)

    def _inp101_out100_viareddim2(self, mat, method_red, method):
        if method_red == 'sum':
            vc = mat.sum(axis=-1)
        elif method_red == 'median':
            vc = mat.median(axis=-1).values
        else:
            raise NotImplementedError

        return self._inp100_out100(vc, method)

    ########## method that take in a "matrix", output (cols_grp1, cols_grp2)
    
    def _inp101_out001(self, mat, method):
        assert mat.ndim == 2
        assert not np.isnan(mat).any()

        if method == 'bipartition_Dh':
            _, _, comps_grp1, comps_grp2 = self._inp101_out101(#torch.abs
                (mat) / np.sum(np.abs(mat), axis=1, keepdims=True),
                method,
                args={'normalize':False}
                )

        elif method == 'svd-sign':
            mat_to_svd = mat #/ np.sqrt(np.sum(np.power(mat, 2),axis=0,keepdim=True))

        elif method == 'svd-sign-rows-L1normalized':
            mat_to_svd = mat / np.sum(np.abs(mat), axis=1, keepdims=True)

        elif method == 'svd-sign-absrows-L2normalized':
            mat_to_svd = np.abs(mat) / np.sqrt(mat.pow(2).sum(dim=0))

        else:
            raise NotImplementedError('requested method has not been implemented in _inp101_out001().')

        
        if method.startswith('svd-sign'):
            assert not np.isnan(mat_to_svd).any()
            svd = np.linalg.svd(mat_to_svd, full_matrices=False)
            comps_grp1 = np.nonzero(svd[2][0,:] < 0)[0].tolist()
            comps_grp2 = self._comps_complement(comps_grp1)

        return None, {'comps_grp1': comps_grp1, 'comps_grp2': comps_grp2}

    ########## method that take in "two matrices", output (inliers, outliers)
    
    def _inp110x2_out100(self, two_mats, method_red, method):
        assert two_mats.shape.tolist() == [self.data.shape[0], self.data.shape[1], 2]
        if method_red == 'last' and method == 'dif':
            _, output = self._inp100_out100(
                two_mats[:, -1, 0] - two_mats[:, -1, 1],
                'MAD'
                )
        else:
            raise NotImplementedError('requested method has not been implemented in _inp110x2_out100().')
        return None, output

    ########## two methods that take in a "3d tensor", output (inliers, outliers) and (cols_grp1,cols_grp2)

    def _inp111_out101_reddim1(self, method_red, method):
        mat = self._reduce_dim1(method_red)
        return self._inp101_out101(mat, method)

    def _inp111_out101(self, data, method):
        assert data.ndim == 3
        assert not np.isnan(data).any()

        if method.startswith('folding122+biclustering(3,3)/folding200:'):
            method_start = 'folding122+biclustering(3,3)/folding200:'
            method_rest = method[len(method_start):]
            _, output100 = self._inp111_out100(data, 'folding200:' + method_rest)
            _, output001 = self._inp111_out001(data[output100['coords_out'], :, :], 'folding122+biclustering(3,3)')
            output = dict(**output100, **output001)
            
        elif method.startswith('folding022+biclustering(3,3)/folding200:'):
            method_start = 'folding022+biclustering(3,3)/folding200:'
            method_rest = method[len(method_start):]
            _, output100 = self._inp111_out100(data, 'folding200:' + method_rest)
            _, output001 = self._inp111_out001(data, 'folding022+biclustering(3,3)')
            output = dict(**output100, **output001)

        elif method.startswith('folding122+biclustering(3,3)/folding100:'):
            method_start = 'folding122+biclustering(3,3)/folding100:'
            method_rest = method[len(method_start):]
            _, output100 = self._inp111_out100(data, 'folding100:' + method_rest)
            _, output001 = self._inp111_out001(data[output100['coords_out'], :, :], 'folding122+biclustering(3,3)')
            output = dict(**output100, **output001)
            
        elif method.startswith('folding022+biclustering(3,3)/folding100:'):
            method_start = 'folding022+biclustering(3,3)/folding100:'
            method_rest = method[len(method_start):]
            _, output100 = self._inp111_out100(data, 'folding100:' + method_rest)
            _, output001 = self._inp111_out001(data, 'folding022+biclustering(3,3)')
            output = dict(**output100, **output001)

        else:
            raise NotImplementedError('requested method has not been implemented in _inp111_out101().')

        return None, output

    ########## method that take in a "3d tensor", output (inliers, outliers)
    
    def _inp111_out100(self, data, method):
        if method == 'median-dim1/rowL2sq/MAD':
            _, output = self._inp100_out100(
                np.power(np.median(data, axis=1), 2).sum(axis=-1),
                'MAD'
                )

        elif method == '2MAD-3majority-1sum':
            coords_out = np.nonzero(np.mean(0.0 + MAD_outlier_det(np.sum(data, axis=2), dm=0, thr=3.5), axis=1) > .5)[0].tolist()
            coords_in = self._coords_in_from_out(coords_out)
            output = {'coords_in': coords_in, 'coords_out': coords_out}


        elif method.startswith('folding200:') or method.startswith('folding100:'):

            # when keeping dim 0, signs do not matter,
            # having a large value (positive or negative) matters. 
            # but interestingly, if we keep the signs (with 'compact' not 'full'),
            # the mat matrix below will be mostly positive <--- good for the .demo()

            if method.startswith('folding200:'):
                method_start = 'folding200:'
                method_rest = method[len(method_start):]
                if 1 == 0:
                    data_ = data
                    print(' not taking abs of 3d data')
                else:
                    data_ = np.abs(data)
                    print(' taking abs of 3d data')

                mat = self._folding(data_, '201') # 201-mat persumably has a planted clique form

            elif method.startswith('folding100:'):
                method_start = 'folding100:'
                method_rest = method[len(method_start):]
                if 1 == 0:
                    data_ = data
                    print(' not taking abs of 3d data')
                else:
                    data_ = np.abs(data)
                    print(' taking abs of 3d data')

                mat = self._folding(data_, '102') # 102-mat persumably has a planted clique form

            n_clusters = 3
            print(' n_clusters :', n_clusters)

            if method_rest == 'SpCo':

                model = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
                model.fit(mat)            
                grps = [(model.row_labels_ == i).nonzero()[0].tolist() for i in range(n_clusters)]

            elif method_rest == 'Agg':

                model = AgglomerativeClustering(n_clusters=n_clusters, compute_full_tree=True)
                model.fit(mat)            
                grps = [(model.labels_ == i).nonzero()[0].tolist() for i in range(n_clusters)]

                '''
                elif method_rest == 'ScipyHi-ward':

                    Z = scipy.cluster.hierarchy.ward(mat)
                    idx = scipy.cluster.hierarchy.leaves_list(Z)
                    fig, ax = plt.subplots(1,1)
                    sns.heatmap(mat[idx,:][:,idx], ax=ax, cmap=my_cmap, center=0)

                elif method_rest == 'ScipyHi-ward-ord':
                    
                    Z = scipy.cluster.hierarchy.ward(mat)
                    idx = scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.optimal_leaf_ordering(Z, mat))
                    fig, ax = plt.subplots(1,1)
                    sns.heatmap(mat[idx,:][:,idx], ax=ax, cmap=my_cmap, center=0)
                '''

            elif method_rest == 'SpBi-log':

                model = SpectralBiclustering(n_clusters=(n_clusters,n_clusters), method="log", n_init=100)
                model.fit(mat)            
                grps = [(model.row_labels_ == i).nonzero()[0].tolist() for i in range(n_clusters)]

            elif method_rest == 'SpBi-bistoch':

                model = SpectralBiclustering(n_clusters=(n_clusters,n_clusters), method="bistochastic", n_init=100)
                model.fit(mat)            
                grps = [(model.row_labels_ == i).nonzero()[0].tolist() for i in range(n_clusters)]

            else:
                raise NotImplementedError('requested method has not been implemented in _inp111_out100() within {0:s}.'.format(method_start))

            grps = [g for g in grps if g!=[]]
            grps_means = [np.mean(mat[:, g][g, :]) for g in grps]
            print(' grps_means:', grps_means)
            coords_out = grps[np.argmax(grps_means)]
            coords_in = self._coords_in_from_out(coords_out)
            output = {'coords_in': coords_in, 'coords_out': coords_out}

            if 1 == 0:#method_rest.startswith('SpBi'):
                print([len(g) for g in grps])
                idx = [i for g in grps for i in g]
                fig, ax = plt.subplots(1,2)
                sns.heatmap(mat, ax=ax[0], cmap=my_cmap, center=0)
                sns.heatmap(mat[idx,:][:,idx], ax=ax[1], cmap=my_cmap, center=0)
                ax[1].set_title('folded matrix; via _inp111_out100, ' + method)
                plt.show()
                  
        else:
            raise NotImplementedError('requested method has not been implemented in _inp111_out100().')

        return None, output
    
    ########## method that take in a "3d tensor", output (cols_grp1,cols_grp2)
    
    def _inp111_out001(self, data, method):

        if method == 'folding122+biclustering(3,3)' or method == 'folding022+biclustering(3,3)':

            # 120-mat and 021-mat below are persumably 3 by 3 checkerboard
            # matrices with only one negative off-diagonal block

            if method == 'folding122+biclustering(3,3)':  # **************
                d_rem = 0
                if 1 == 0:##was 1
                    data_ = data
                else:
                    data_ = np.divide(
                        data,
                        np.sqrt(np.sum(np.power(data, 2), axis=d_rem, keepdims=True)) + 1e-6
                        )
                    print(' normalized 3d data by L2 along dim ', d_rem)
                mat = self._folding(data_, '120') 
            elif method == 'folding022+biclustering(3,3)':
                d_rem = 1
                if 1 == 0:
                    data_ = data
                else:
                    data_ = np.divide(
                        data,
                        np.sqrt(np.sum(np.power(data, 2), axis=d_rem, keepdims=True)) + 1e-6
                        )
                    print(' normalized 3d data by L2 along dim ', d_rem)
                mat = self._folding(data_, '021')

            n_clusters = 3
            model = SpectralBiclustering(
                n_clusters=(n_clusters,n_clusters),
                method="bistochastic",
                random_state=0)
            if 1 == 1:
                model.fit(np.sign(mat))
                print(' taking sign(mat)')
                # taking the sign might lead to not having 3 clusters
            else:
                model.fit(np.sign(mat))
                print(' not taking sign(mat)')
            grps = [(model.row_labels_ == i).nonzero()[0].tolist() for i in range(n_clusters)]
            print(grps)
            grps = [g for g in grps if g!=[]]
                    

            if len(grps) == 3:
                # among the 3 clusters, we look for the two whose corresponding
                # off-diagonal block has the smallest mean (our model says this will be negative)
                scr = []
                for ii in range(n_clusters):
                    for jj in range(ii+1,n_clusters):
                        scr += [(ii, jj, np.mean(mat[grps[ii],:][:,grps[jj]]))] 
                ids = scr[np.argmin(np.array([s[2] for s in scr]))][:2]
                print(scr,ids)
            elif len(grps) == 2:
                ids = [0, 1]
                
            comps_grp1 = grps[ids[0]]
            comps_grp2 = grps[ids[1]]
            
            if 1 == 0:
                idx = [i for g in grps for i in g]
                fig, ax = plt.subplots(1,2)
                sns.heatmap(mat[idx,:][:,idx], ax=ax[0], cmap=my_cmap, center=0)
                sns.heatmap(np.sign(mat)[idx,:][:,idx], ax=ax[1], cmap=my_cmap, center=0)
                fig.suptitle('folded matrix; via _inp111_out001, ' + method)
                plt.show()

        else:
            raise NotImplementedError('requested method has not been implemented in _inp111_out001().')

        return None, {'comps_grp1': comps_grp1, 'comps_grp2': comps_grp2}

##################################################################################################################
##################################################################################################################

class TriClassificationInstances(TriClassification):
    def __init__(self, data, weights):
        """
        Methods:
        --------
            get_list_instructions_inp111()
                generates a list of classification instructions
            compute_instr_outputs()
                computes the classification output for each instruction from the list
            plot_list_triclassifications(),
                for each 'output' dictionary,
                plots coordinates and components (whichever is provided)
                over the 'median-over-steps matrix'.
                (TODO) plots several folded matrices (200, 022, etc, depending on which groupings are available in the dictionary) after permuting the row and column according to the groups
        """
        super(TriClassificationInstances, self).__init__(data, weights)
        self.experiments = []
        self.compute_instr_outputs()

    ##########

    def get_list_instructions_inp111_out101(self):
        return [[('_inp111_out101', 'folding122+biclustering(3,3)/folding200:SpCo')],
                [('_inp111_out101', 'folding122+biclustering(3,3)/folding200:Agg')],
                #[('_inp111_out101', 'folding122+biclustering(3,3)/folding200:ScipyHi-ward')],
                #[('_inp111_out101', 'folding122+biclustering(3,3)/folding200:ScipyHi-ward-ord')],
                [('_inp111_out101', 'folding122+biclustering(3,3)/folding200:SpBi-log')],
                [('_inp111_out101', 'folding122+biclustering(3,3)/folding200:SpBi-bistoch')],
                #
                [('_inp111_out101', 'folding022+biclustering(3,3)/folding200:SpCo')],
                [('_inp111_out101', 'folding022+biclustering(3,3)/folding200:Agg')],
                #[('_inp111_out101', 'folding022+biclustering(3,3)/folding200:ScipyHi-ward')],
                #[('_inp111_out101', 'folding022+biclustering(3,3)/folding200:ScipyHi-ward-ord')],
                [('_inp111_out101', 'folding022+biclustering(3,3)/folding200:SpBi-log')],
                [('_inp111_out101', 'folding022+biclustering(3,3)/folding200:SpBi-bistoch')],
                #
                [('_inp111_out101', 'folding122+biclustering(3,3)/folding100:SpCo')],
                [('_inp111_out101', 'folding122+biclustering(3,3)/folding100:Agg')],
                #[('_inp111_out101', 'folding122+biclustering(3,3)/folding100:ScipyHi-ward')],
                #[('_inp111_out101', 'folding122+biclustering(3,3)/folding100:ScipyHi-ward-ord')],
                [('_inp111_out101', 'folding122+biclustering(3,3)/folding100:SpBi-log')],
                [('_inp111_out101', 'folding122+biclustering(3,3)/folding100:SpBi-bistoch')],
                #
                [('_inp111_out101', 'folding022+biclustering(3,3)/folding100:SpCo')],
                [('_inp111_out101', 'folding022+biclustering(3,3)/folding100:Agg')],
                #[('_inp111_out101', 'folding022+biclustering(3,3)/folding100:ScipyHi-ward')],
                #[('_inp111_out101', 'folding022+biclustering(3,3)/folding100:ScipyHi-ward-ord')],
                [('_inp111_out101', 'folding022+biclustering(3,3)/folding100:SpBi-log')],
                [('_inp111_out101', 'folding022+biclustering(3,3)/folding100:SpBi-bistoch')],
                #
                [('_inp111_data101', 'median'), ('_inp101_out101', 'coords:L1-MAD/comps:svd-sign')], # triclustering(mags) used to perform this
                # group_coords_2comps_biclustering() used to plot these:
                [('_inp111_data101', 'last'), ('_inp101_data101', 'normalize_rows_byL1'), ('_inp101_out101', 'bipartition_Dh_normalizeFalse')],
                [('_inp111_data101', 'last'), ('_inp101_out101', 'bipartition_spectral')],
                [('_inp111_data101', 'median'), ('_inp101_data101', 'normalize_rows_byL1'), ('_inp101_out101', 'bipartition_Dh_normalizeFalse')],
                [('_inp111_data101', 'median'), ('_inp101_out101', 'bipartition_spectral')],
                ]

    def get_list_instructions_inp111_out100(self):
        return [[('_inp111_out100', 'median-dim1/rowL2sq/MAD')],
                [('_inp111_out100', '2MAD-3majority-1sum')], ## outlier detection via '1-d trajectories' (agg all comps to 1 dim, id the outlier at each time according to 1d aggregation, for each coord take the majority over time to declare ass outlier)
                #
                [('_inp111_out100', 'folding200:SpCo')],
                [('_inp111_out100', 'folding200:Agg')],
                #[('_inp111_out100', 'folding200:ScipyHi-ward')],
                #[('_inp111_out100', 'folding200:ScipyHi-ward-ord')],
                [('_inp111_out100', 'folding200:SpBi-log')],
                [('_inp111_out100', 'folding200:SpBi-bistoch')],
                #
                [('_inp111_out100', 'folding100:SpCo')],
                [('_inp111_out100', 'folding100:Agg')],
                #[('_inp111_out100', 'folding100:ScipyHi-ward')],
                #[('_inp111_out100', 'folding100:ScipyHi-ward-ord')],
                [('_inp111_out100', 'folding100:SpBi-log')],
                [('_inp111_out100', 'folding100:SpBi-bistoch')],
                #
                [('_inp111_data101', 'median'), ('_inp101_data100', 'rows_L2sq'), ('_inp100_out100', 'MAD')]
                ]

    def get_list_instructions_inp111_out001(self):
        return [[('_inp111_out001', 'folding122+biclustering(3,3)')],
                [('_inp111_out001', 'folding022+biclustering(3,3)')],
                #
                [('_inp111_data101', 'last'), ('_inp101_out001', 'svd-sign')],
                [('_inp111_data101', 'last'), ('_inp101_data101', 'normalize_rows_byL1'), ('_inp101_out001', 'svd-sign')],
                [('_inp111_data101', 'last'), ('_inp101_data101', 'normalize_cols_byL2'), ('_inp101_out001', 'svd-sign')],
                [('_inp111_data101', 'median'), ('_inp101_out001', 'svd-sign')], 
                [('_inp111_data101', 'median'), ('_inp101_data101', 'normalize_rows_byL1'), ('_inp101_out001', 'svd-sign')],
                [('_inp111_data101', 'median'), ('_inp101_data101', 'normalize_cols_byL2'), ('_inp101_out001', 'svd-sign')],
                ]

    def get_list_instructions_inp111(self):
        return (self.get_list_instructions_inp111_out101() +
                self.get_list_instructions_inp111_out100() +
                self.get_list_instructions_inp111_out001())#[:3]

    ##########
    
    def compute_instr_outputs(self):
        list_instructions = self.get_list_instructions_inp111()
        print('\n\n------------ computing the classification outputs for {0:d} instructions lists:'.format(len(list_instructions)))
        for i_instr, instr in enumerate(list_instructions):
            print('\n')
            print('[compute_instr_outputs()] executing instruction set {0} of range({1})'.format(i_instr, len(list_instructions)))
            
            gr = Groups(self.classify(instr))
            if gr.has_comps_keys() and gr.has_coords_keys():
                src = 'TCI-comps-coords'
            elif gr.has_comps_keys() and not gr.has_coords_keys():
                src = 'TCI-comps'
            elif not gr.has_comps_keys() and gr.has_coords_keys():
                src = 'TCI-coords'
            else:
                src = 'TCI-empty'

            self.experiments += [
                Experiment(source=src,
                           method=self._instr_to_str(instr),
                           detail='',
                           groups=gr,
                           scores=(None,None)
                           )
                ]
            
            print(self.experiments[-1])
            
        return

    def plot_list_triclassifications(self):

        print('\n\n------------ running plot_list_triclassifications() ; plotting computed outputs for our {0:d} lists of instructions, on the matrix of median-over-time'.format(
            len(self.experiments)))

        num_cols = 4
        num_rows = len(self.experiments)//num_cols + (len(self.experiments)%num_cols > 0)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20,5*num_rows))
        if num_rows == 1:
            axs = [axs]
        ax_cnt = -1
        
        mat_, _ = self._inp111_data101(self.data, 'median') ## why is this inside the loop ??? 

        for ii, exp_ in enumerate(self.experiments):
            print(ii)

            instr = getattr(exp_, 'detail')
            output = getattr(exp_, 'groups').groups

            ax_cnt += 1
            ax_row = ax_cnt//num_cols
            ax_col = ax_cnt%num_cols
            ax = axs[ax_row][ax_col]

            mat = mat_.copy()
            
            if 'coords_in' in output and 'coords_out' in output:
                id_rows = output['coords_in'] + output['coords_out']
                mat = mat[id_rows, :]
                ax.axhline(len(output['coords_in']), color='red')
                ticks_y = [i for i in range(0, len(id_rows), 1 + len(id_rows) // 15)]
                ax.set_yticks(np.array(ticks_y)+0.5)
                ax.set_yticklabels([id_rows[i] for i in ticks_y])

            if 'comps_grp1' in output and 'comps_grp2' in output:
                id_cols = output['comps_grp1'] + output['comps_grp2']
                mat = mat[:, id_cols]
                ax.axvline(len(output['comps_grp1']), color='red')
                ticks_x = [i for i in range(0, len(id_cols), 1 + len(id_cols) // 15)]
                ax.set_xticks(np.array(ticks_x) + 0.5)
                ax.set_xticklabels([id_cols[i] for i in ticks_x])

            ax.imshow(mat, interpolation=None, cmap=my_cmap)

            ax.set_title(instr,size=5)

        print('------------ finished running plot_list_triclassifications()')

        return fig, axs