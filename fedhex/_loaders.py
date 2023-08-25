"""
Classes that load data.
"""

from numpy import ndarray, save
from typing import Callable

from .io import DEFAULT_cut, DEFAULT_exps, evt_sel_1, find_root, load_data_dict, load_numpy, load_root
from .utils import LOG_ERROR, print_msg

from ._managers import DataManager


class Loader(DataManager):

    def __init__(self, path: str|None=None, data_dict: dict|None=None):
        """
        Instantiate a Loader with optional arguments for the path where data
        are located and the dictionary of data.
        """
        super().__init__()
        self._path = path
        self._data_dict = data_dict

    @property
    def path(self) -> str:
        return self._path
    
    @property
    def thresh(self) -> int:
        return self._thresh

    def load(self) -> tuple[ndarray, ndarray]:
        """
        Loads the data from a given path/data_dict. Updates the Loader's path
        or data_dict variables if provided.
        """

        if self._data_dict is None and self._path is not None:
            self._data_dict = load_data_dict(self._path, ret_dict=True)

        if self._data_dict is None and self._path is None:
            print_msg("Loader cannot have `None` for both `path` and " + \
                      "`data_dict`. Please provide values for either",
                      level=LOG_ERROR)
        
        self._data = self._data_dict.get("data")
        self._cond = self._data_dict.get("cond")
        self._whiten_data = self._data_dict.get("whiten_data")
        self._whiten_cond = self._data_dict.get("whiten_cond")
        self.has_preprocessed = True

        return self.recover()
    
    def save_to_npy(self, path_npy: str):
        save(path_npy, self._data_dict, allow_pickle=True)

    def save_to_root(self):
        """
        Darshan's ROOT saving tool (uses self.path since it points to .ROOT)
        """
        pass


class NumpyLoader(Loader):

    def __init__(self, path: str, path_labels: str, data_dict: dict=None):
        """
        path : str
            relevant .ROOT path for data loaded or saved.
        """
        super().__init__(path=path, data_dict=data_dict)
        self._path_labels = path_labels
        
    def load(self, event_thresh: int=0) -> tuple[ndarray, ndarray]:

        self._thresh = event_thresh
        self._samples, self._labels = load_numpy(self._path, event_thresh=event_thresh)
        self.has_original = True

        return self._samples, self._labels


class RootLoader(Loader):

    def __init__(self, path: str, data_dict: dict=None):
        """
        path : str
            relevant .ROOT path for data loaded or saved.
        """
        super().__init__(path=path, data_dict=data_dict)
        
    def load(self, event_thresh: int=0, max_depth: int=3,
             event_selection_fn: Callable[[ndarray], tuple[ndarray]]=evt_sel_1,
             cutstr: str=DEFAULT_cut, exps: str=DEFAULT_exps) -> tuple[ndarray, ndarray]:

        self._thresh = event_thresh

        file_list = find_root(self._path, max_depth=max_depth)
        self._samples, self._labels = load_root(file_list,
            event_selection_fn=event_selection_fn, expressions=exps,
            cutstr=cutstr, event_thresh=event_thresh)
        self.has_original = True
        
        return self._samples, self._labels
