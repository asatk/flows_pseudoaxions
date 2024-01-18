"""
Classes that load data.
"""

from numpy import ndarray, save
from typing import Self

from .io import find_root, load_config, load_data_dict, load_numpy, load_root, \
    save_root
from .utils import LOG_ERROR, print_msg

from ._managers import DataManager


class Loader(DataManager):

    @classmethod
    def import_cfg(cls: Self, config_path: str) -> Self:
        
        state_dict = load_config(config_path)

        kw_list = [
            "path"
        ]

        missing_kws = set(kw_list) - set(state_dict)
        if missing_kws != set():
            print_msg(f"Loaded config at `{config_path}` does not contain " + \
                      "the necessary kws for this `Loader:\n" + \
                      f"{missing_kws}", level=LOG_ERROR)
            return None
        
        path = state_dict.get("path")
        obj = Loader(path=path, data_dict=None)
        obj._set_state(state=state_dict)

        return obj
    
    def _set_state(self, state: dict) -> None:
        if "path" in state:
            self._path = state.get("path")
        self.state_dict = state

    def __init__(self, path: str|None=None, data_dict: dict|None=None):
        """
        Instantiate a Loader with optional arguments for the path where data
        are located and the dictionary of data.
        """
        super().__init__()
        self._path = path
        self._data_dict = data_dict
        self._thresh = 0

        self.state_dict.update({
            "load_path": self._path,
        })

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
        self.state_dict.update({"save_path_npy": path_npy})
        save(path_npy, self._data_dict, allow_pickle=True)

    def save_to_root(self, save_root_path: str, custom:bool = True):
        """
        save_root_path: str
            path to create .ROOT to be saved to, .ROOT added if not present
            
        custom: bool
            if True, the keys of the data_dict will be the branches 
            of the TTree and the corresponding values of the dict
            will populate those branches.
            
            if False, the data_dict must be in the format of 
            {"gen_samples": gen_samples, "gen_labels": gen_labels, "trn_samples": samples, "trn_labels": labels}.
            Training samples/labels are optional. Labels must be repeated to be the same dimensions as samples.
        """
        self.state_dict.update({"save_root_path": save_root_path})
        save_root(save_root_path, self._data_dict, custom)


class NumpyLoader(Loader):

    def __init__(self, path: str, path_labels: str, data_dict: dict=None):
        """
        path : str
            relevant .ROOT path for data loaded or saved.
        """
        super().__init__(path=path, data_dict=data_dict)
        self._path_labels = path_labels

    def _set_state(self, state: dict) -> None:
        if "path" in state:
            self._path = state.get("path")
        if "path_labels" in state:
            self._path_labels = state.get("path_labels")
        if "thresh" in state:
            self._thresh = state.get("thresh", 0)
        self.state_dict = state
        
    def load(self, event_thresh: int=0) -> tuple[ndarray, ndarray]:

        self._thresh = event_thresh
        self._samples, self._labels = load_numpy(self._path, event_thresh=event_thresh)
        self.has_original = True

        return self._samples, self._labels


class RootLoader(Loader):

    @classmethod
    def import_cfg(cls: Self, config_path: str) -> Self:
        
        state_dict = load_config(config_path)

        kw_list = [
            "path",
            "thresh",
            "max_depth",
            "event_selection_function",
            "cutstr",
            "exps",
            "file_list"
        ]

        missing_kws = set(kw_list) - set(state_dict)
        if missing_kws != set():
            print_msg(f"Loaded config at `{config_path}` does not contain " + \
                      "the necessary kws for this `RootLoader:\n" + \
                      f"{missing_kws}", level=LOG_ERROR)
            return None
        
        path = state_dict.get("path")
        obj = RootLoader(path=path, data_dict=None)
        obj._set_state(state=state_dict)

        # self.state_dict = state_dict
        # self._path = state_dict.get("path")

        return obj


    def __init__(self, path: str, data_dict: dict=None):
        """
        path : str
            relevant .ROOT path for data loaded or saved.
        """
        super().__init__(path=path, data_dict=data_dict)

        # TODO add config param and add all relevant kws to obj
        
    @property
    def file_list(self):
        return self._file_list

    def _set_state(self, state: dict) -> None:
        if "path" in state:
            self._path = state.get("path")
        if "thresh" in state:
            self._thresh = state.get("thresh", 0)
        if "file_list" in state:
            self._file_list = state.get("file_list")
        self.state_dict = state
    
    def load(self,
             tree_name: str=None,
             data_vars: list[str]=None,
             cond_vars: list[str]=None,
             cutstr: str=None,
             defs: dict[str, str]=None,
             file_list: str|list[str]=None,
             event_thresh: int=0,
             max_depth: int=3) -> tuple[ndarray, ndarray]:

        if tree_name is None or \
                data_vars is None or \
                cond_vars is None or \
                cutstr is None or \
                defs is None:
            print("Necessary ROOT loading parameters not provided.")
            return (None, None)

        self._thresh = event_thresh

        if file_list is None:
            self._file_list = find_root(self._path, max_depth=max_depth)
        else:
            self._file_list = file_list
        
        self._samples, self._labels = load_root(
            root_dir=self._file_list,
            tree_name=tree_name,
            data_vars=data_vars,
            cond_vars=cond_vars,
            defs=defs,
            cutstr=cutstr,
            event_thresh=event_thresh,
            max_depth=max_depth)
        self.has_original = True

        self.state_dict.update({
            "path": self._path,
            "thresh": self._thresh,
            "max_depth": max_depth,
            "cutstr": cutstr,
            "data_vars": data_vars,
            "cond_vars": cond_vars,
            "defs": defs,
            "file_list": self._file_list,
            "tree_name": tree_name
        })
        
        return self._samples, self._labels
