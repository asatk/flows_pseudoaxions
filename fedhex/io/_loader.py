from numpy import ndarray, save

from . import load_data_dict
from ._root import _loadallroot
from ..pretrain import dewhiten, whiten


class Loader():

    def __init__(self, path: str, data_dict: dict|None=None):
        self.path = path

        if data_dict is None:
            data_dict = self.__load(path)

        self.data_dict = data_dict
        
        # TODO try/except
        self.data = data_dict["data"]
        self.cond = data_dict["cond"]
        self.whiten_data = data_dict["whiten_data"]
        self.whiten_cond = data_dict["whiten_cond"]

    def __str__(self):
        s = f"<{self.__class__.__name__}>: "
        s += str(self.__dict__)
        return s

    def get_data_dict(self):
        return self.data_dict
    
    def get_data(self):
        return self.data
    
    def get_cond(self):
        return self.cond
    
    def get_whiten_data(self):
        return self.whiten_data
    
    def get_whiten_cond(self):
        return self.whiten_cond
    
    def get_path(self):
        return self.path

    def __load(self, path: str) -> dict:
        return load_data_dict(path, ret_dict=True)

    def recover_preproc(self) -> tuple[ndarray, ndarray]:
        samples = dewhiten(self.data, self.whiten_data)
        labels = dewhiten(self.cond, self.whiten_cond)
        return samples, labels
    
    def recover_new(self, samples, is_cond: bool=False) -> ndarray:
        whiten_data = self.whiten_cond if is_cond else self.whiten_data
        return dewhiten(samples, whiten_data)

    def preproc_new(self, samples, is_cond: bool=False) -> ndarray:
        whiten_data = self.whiten_cond if is_cond else self.whiten_data
        return whiten(samples, whiten_data)
    
class RootLoader(Loader):

    def __init__(self, path: str, data_dict: dict=None, event_threshold: float=0.01):
        """
        path : str
            relevant .ROOT path for data loaded or saved.
        """
        self.thresh = event_threshold
        if data_dict is None:
            data_dict = self.__load(path, event_threshold=event_threshold)
        
        super().__init__(path, data_dict=data_dict)
        
    def __load(self, path: str, event_threshold: float=0.01) -> dict:
        samples, labels = _loadallroot(path, event_threshold=event_threshold)
        data, whiten_data = whiten(samples)
        cond, whiten_cond = whiten(labels)

        data_dict = {
            "data": data,
            "cond": cond,
            "whiten_data": whiten_data,
            "whiten_cond": whiten_cond
        }

        return data_dict

    def save_to_npy(self, path_npy: str):
        save(path_npy, self.data_dict, allow_pickle=True)

    def save_to_root(self):
        """
        Darshan's ROOT saving tool (uses self.path since it points to .ROOT)
        """
        pass