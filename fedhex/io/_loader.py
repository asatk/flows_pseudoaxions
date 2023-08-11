from numpy import ndarray, save

from ._data import load_data_dict
from ._root import _evt_sel_1, _find_root, _load_root, cutstr, expressions
from ..pretrain import dewhiten, whiten
from ..utils import LOG_ERROR, print_msg
from ..constants import WHITEN_EPSILON


class Loader():

    def __init__(self, path: str|None=None, data_dict: dict|None=None):
        """
        Instantiate a Loader with optional arguments for the path where data
        are located and the dictionary of data.
        """

        self._path = path
        self._data_dict = data_dict

    def __str__(self):
        s = f"<{self.__class__.__name__}>: "
        s += str(self.__dict__)
        return s

    def get_path(self):
        return self._path

    def get_data_dict(self):
        return self._data_dict
    
    def get_data(self):
        return self._data
    
    def get_cond(self):
        return self._cond
    
    def get_whiten_data(self):
        return self._whiten_data
    
    def get_whiten_cond(self):
        return self._whiten_cond

    def load(self, path: str|None=None, data_dict: dict|None=None,
             epsilon: float=WHITEN_EPSILON) -> dict:
        """
        Loads the data from a given path/data_dict. Updates the Loader's path
        or data_dict variables if provided.
        """

        if path is not None:
            self._path = path
        
        if data_dict is not None:
            self._data_dict = data_dict

        if self._data_dict is None and self._path is not None:
            self._data_dict = data_dict = load_data_dict(path, ret_dict=True)

        if self._data_dict is None and self.__path__ is None:
            print_msg("Loader cannot have `None` for both `path` and " + \
                      "`data_dict`. Please provide values for either", level=LOG_ERROR)
        
        # TODO try/except or get w/ default val
        self._data = data_dict.get("data")
        self._cond = data_dict.get("cond")
        self._whiten_data = data_dict.get("whiten_data")
        self._whiten_cond = data_dict.get("whiten_cond")

        return data_dict

    def recover_preproc(self) -> tuple[ndarray, ndarray]:
        samples = dewhiten(self._data, self._whiten_data)
        labels = dewhiten(self._cond, self._whiten_cond)
        return samples, labels
    
    def recover_new(self, samples, is_cond: bool=False) -> ndarray:
        whiten_data = self._whiten_cond if is_cond else self._whiten_data
        return dewhiten(samples, whiten_data)

    def preproc_new(self, samples, is_cond: bool=False) -> ndarray:
        whiten_data = self._whiten_cond if is_cond else self._whiten_data
        return whiten(samples, whiten_data)
    
class RootLoader(Loader):

    def __init__(self, path: str, data_dict: dict=None, event_thresh: int=100):
        """
        path : str
            relevant .ROOT path for data loaded or saved.
        """
        super().__init__(path=path, data_dict=data_dict)
        self._thresh = event_thresh
        
    def load(self, path: str|None=None, event_thresh: int|None=None,
             epsilon: float=WHITEN_EPSILON) -> dict:

        if path is not None:
            self._path = path
        else:
            path = self._path

        if event_thresh is not None:
            self._thresh = event_thresh
        else:
            event_thresh = self._thresh

        print(path)
        file_list = _find_root(path)
        samples, labels = _load_root(file_list, event_selection_fn=_evt_sel_1,
            expressions=expressions, cutstr=cutstr, event_thresh=event_thresh)
        data, whiten_data = whiten(samples, epsilon=epsilon)
        cond, whiten_cond = whiten(labels, epsilon=epsilon)

        data_dict = {
            "data": data,
            "cond": cond,
            "whiten_data": whiten_data,
            "whiten_cond": whiten_cond
        }

        return super().load(path=self._path, data_dict=data_dict)

    def save_to_npy(self, path_npy: str):
        save(path_npy, self._data_dict, allow_pickle=True)

    def save_to_root(self):
        """
        Darshan's ROOT saving tool (uses self.path since it points to .ROOT)
        """
        pass