import abc
import os
import shutil
import pickle
import copy

import torch

from ccci.utils import class_str


class DataLoaderBase(metaclass=abc.ABCMeta):
    """Abstract class for downloading checkpoints
    and creating magnitudes tensor (the CCC-data), across
    coordinates, checkpoints, and components
    """

    def __init__(
        self,
        model_name,
        model_args,
        steps_list
        ):

        self.model_name = model_name
        self.model_args = model_args
        self.steps_list = steps_list

        self._initialized = False

        self.modes_list = []
        self.traj_info_nd = {}

        self.cache_dir = None

        print('Instantiated ', self.__class__.__name__)

    def __str__(self):
        return class_str(
            self.__class__.__name__, 
            [(attr, getattr(self, attr)) for attr in ['model_name']] +
            [('n_ckpt', len(self.steps_list))]
            )

    @abc.abstractmethod
    def get_ckpt(self, step):
        """Download/load a model"""
        pass

    def delete_cached(self):
        if os.path.isdir(self.cache_dir):
            try:
                shutil.rmtree(self.cache_dir)
                print('shutil.rmtree : ', self.cache_dir)
            except Exception as e:
                print('Could not delete the cache directory. Reason:', e)

    @abc.abstractmethod
    def set_modes_list(self):
        """Set the list of aggregation modes to be computed
        for each checkpoint.
        """
        pass

    @abc.abstractmethod
    def get_comps_rownorms(model, model_args, mode):
        """Given a model, produce component names, magnitudes, and sizes.

        Any number of "modes" will have to be implemented here;
        e.g., 'full', 'compact', etc
        """
        pass

    def model_to_mags(self, model, mode):
        """Convert the model to weight magnitudes tensor."""
        print('running model_to_mags with mode', mode)
        comps = self.get_comps_rownorms(model, mode)

        names = [tt[0] for tt in comps]
        mags = torch.cat([tt[1][:, None] for tt in comps], dim=1)
        lengths = [tt[2] for tt in comps]

        assert self.model_args['n_embd'] == mags.shape[0]
        assert len(names) == len(lengths)
        assert len(names) == mags.shape[1]

        return names, mags, lengths

    def init_modes(self, modes_list, step_ref):
        """Initialize class attributes (such as sizes of tensors, etc)
        by downloading one checkpoint.
        """
        print('running init_modes:')
        if not self._initialized:

            assert step_ref in self.steps_list
            j_ref = self.steps_list.index(step_ref)
            model = self.get_ckpt(step_ref)

            for mode in modes_list:
                if mode not in self.traj_info_nd:
                    names, mags, lengths = self.model_to_mags(model, mode)
                    self.traj_info_nd[mode] = {}
                    self.traj_info_nd[mode]['num'] = len(names)
                    self.traj_info_nd[mode]['names'] = names
                    self.traj_info_nd[mode]['lengths'] = lengths
                    self.traj_info_nd[mode]['mags_set'] = [
                        False for tt in range(len(self.steps_list))
                        ]
                    self.traj_info_nd[mode]['mags'] = torch.zeros(
                        self.model_args['n_embd'],
                        len(self.steps_list),
                        len(names)
                        )
                    self.traj_info_nd[mode]['mags'][:, j_ref, :] = mags
                    self.traj_info_nd[mode]['mags_set'][j_ref] = True

            self._initialized = True
        self.delete_cached()
        print('done with init_modes.')
        return

    @abc.abstractmethod
    def set_path(self):
        """Set the path for saving the class instance to disk."""
        pass

    def save(self):
        """saves the class instance to disk."""
        filename = os.path.join(self.path, self.model_name + '.pt')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as file_:
            pickle.dump(self, file_, -1)
            print('saved as', filename)

    def compute_all(self, modes_list):
        """Compute the aggregate weights magnitudes
        according to the modes given in modes_list.

        Parameters
        ----------
        modes_list : List[str]
            A list of modes for recording aggregate weights magnitudes.
        """
        print('running compute_all:')
        to_dl = False
        for mode in modes_list:
            to_dl = to_dl or not all(self.traj_info_nd[mode]['mags_set'])
        if to_dl:
            print('some downloading is required.')
            # i.e. there exists a mode and a step for which download is needed
            for j, step in enumerate(self.steps_list):
                print('---- (checkpoint {0} of {1}) step : {2}'.format(
                    j, len(self.steps_list)-1, step))

                to_dl = False
                for mode in modes_list:
                    to_dl = to_dl or not self.traj_info_nd[mode]['mags_set'][j]
                if to_dl:
                    # i.e. there exists a mode for which download is needed for this step
                    model = self.get_ckpt(step)

                    for mode in modes_list:
                        print(
                            'self.traj_info_nd[{0}][mags_set][{1}] :'.format(mode, j),
                            self.traj_info_nd[mode]['mags_set'][j]
                            )

                        if not self.traj_info_nd[mode]['mags_set'][j]:
                            names, mags, lengths = self.model_to_mags(model, mode)
                            self.traj_info_nd[mode]['mags'][:, j, :] = mags
                            self.traj_info_nd[mode]['mags_set'][j] = True

                    del model # free up memory
                    self.delete_cached()
                    self.save()

                else:
                    print('all requested modes already exist at this step.')
        else:
            print('all requested modes already exist at all steps.')
        print('done with compute_all.')

    def remove_steps(self, steps_to_remove):
        """Remove a given step from the steps dimension of all attributes."""
        steps_to_remove = [i for i in steps_to_remove if i in self.steps_list]
        if steps_to_remove:
            d = {step: idx for idx, step in enumerate(self.steps_list)}
            ids_to_remove = [d.get(step) for step in steps_to_remove]
            ids_to_keep = [i for i in range(len(self.steps_list))
                           if i not in ids_to_remove]

            self.steps_list = [
                s for s in self.steps_list
                if s not in steps_to_remove
                ]  # maintains ordering
            for mode in self.traj_info_nd:
                self.traj_info_nd[mode]['mags_set'] = [
                    s for i, s in enumerate(self.traj_info_nd[mode]['mags_set'])
                    if i in ids_to_keep]
                self.traj_info_nd[mode]['mags'] = self.traj_info_nd[mode]['mags'][:, ids_to_keep, :]

    def _eq_traj_info_nd(self, other):
        """Determine if self and other are equal in
        their traj_info_nd attribute.
        """
        is_eq = True
        if self.modes_list != other.modes_list:
            return False
        else:
            for mode in self.modes_list:
                for k in self.traj_info_nd[mode]:
                    if torch.is_tensor(self.traj_info_nd[mode][k]):
                        is_eq = (
                            is_eq and torch.equal(
                                self.traj_info_nd[mode][k],
                                other.traj_info_nd[mode][k]
                                )
                                )
                    else:
                        is_eq = (
                            is_eq
                            and self.traj_info_nd[mode][k] == other.traj_info_nd[mode][k]
                            )
                    if not is_eq:
                        return False
        return True

    def __eq__(self, other):
        """Determine if self and other are equal."""
        return all([
            self.model_name == other.model_name,
            self.model_args == other.model_args,
            self.steps_list == other.steps_list,
            self._initialized == other._initialized,
            self.modes_list == other.modes_list,
            self._eq_traj_info_nd(other)],
            )

    @abc.abstractmethod
    def process(self):
        """Applying any necessary processing; e.g., removing some steps, etc."""

    def prepare(self):
        """Download the checkpoints and creating the CCC-data tensor,
        or load if all requested 'modes'' already eists on disk.
        """
        
        ## data:

        # TODO : https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
        try:
            fname = os.path.join(self.path, self.model_name + '.pt')
            print(os.path.isfile(fname))
            with open(fname, "rb") as file_:
                print('successfully opened the file ', fname)
                C = pickle.load(file_)
                
                print('self.__dict__.keys():', self.__dict__.keys())
                print('C.__dict__.keys():', C.__dict__.keys())
                print(C.steps_list)
                self.__dict__ = copy.deepcopy(C.__dict__)
                #self.__dict__.update(C.__dict__)
                print(self)

        except Exception as e:
            print(
                'Could not load CCC-data for model {0} from disk. Reason:'.format(
                    self.model_name
                    ), e
                    )
        
        ## apply any necessary clean up and processing:
        self.process()

        ## initialize necessary attributes using the last checkpoint:
        self.init_modes(self.modes_list, self.steps_list[-1])

        ## perform all necessary computations for all checkpoints:
        self.compute_all(self.modes_list)

        return

