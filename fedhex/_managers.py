import abc
from numpy import ndarray

from .constants import WHITEN_EPSILON
from .io import save_config
from .pretrain import dewhiten, whiten
from .posttrain import intersect_labels
from .utils import LOG_ERROR, print_msg


class DataManager(metaclass=abc.ABCMeta):
    
    def __init__(self):
        self.has_preprocessed = False
        self.has_original = False

    def __str__(self):
        return f"<<{self.__class__.__name__}>>"
    
    # @property
    # def has_preprocessed(self) -> bool:
    #     return self._has_preprocessed
    
    # @property
    # def has_original(self) -> bool:
    #     return self._has_original

    @property
    def samples(self) -> ndarray:
        return self._samples
    
    @property
    def labels(self) -> ndarray:
        return self._labels

    @property
    def data(self) -> ndarray:
        return self._data
    
    @property
    def cond(self) -> ndarray:
        return self._cond
    
    @property
    def whiten_data(self) -> dict:
        return self._whiten_data
    
    @property
    def whiten_cond(self) -> dict:
        return self._whiten_cond
    
    @property
    def data_dict(self) -> dict:
        return self._data_dict

    def save(self, config_path: str) -> bool:

        prefix = f"_{self.__class__.__name__}"
        len_prefix = len(prefix)
        
        d = {k: v for k, v in self.__dict__.items() if k[0] != "_"}
        d.update({k[len_prefix:]: v for k, v in self.__dict__.items() if k[len_prefix] == prefix})
        
        return save_config(config_path, d)
    
    def preproc(self, epsilon: float=WHITEN_EPSILON) -> tuple[ndarray, ndarray]:
        if not self.has_preprocessed:
            if not self.has_original:
                print_msg("For a <data, conditional data> pair to be " + \
                          "returned from `preproc()`, either this instance "+\
                          "of `DataManager` already has access to these " + \
                          "data OR it must have access to the unprocessed " + \
                          "<samples, labels> pair.",
                          level=LOG_ERROR)
                return tuple()
            
            self._data, self._whiten_data = whiten(self._samples,
                epsilon=epsilon, ret_dict=True)
            self._cond, self._whiten_cond = whiten(self._labels,
                epsilon=epsilon, ret_dict=True)

            self._data_dict = {
                "data": self._data,
                "cond": self._cond,
                "whiten_data": self._whiten_data,
                "whiten_cond": self._whiten_cond
            }

            self.has_preprocessed = True
        
        return self._data, self._cond

    def recover(self) -> tuple[ndarray, ndarray]:
        if not self.has_original:
            if not self.has_preprocessed:
                print_msg("For a <samples, labels> pair to be returned " + \
                          "from `recover()`, either this instance of " + \
                          "`DataManager` already has access to these data " + \
                          "OR it must have access to the preprocessed " + \
                          "<data, conditional data> pair.",
                          level=LOG_ERROR)
                return tuple()
            
            self._samples = dewhiten(self._data, self._whiten_data)
            self._labels = dewhiten(self._cond, self._whiten_cond)
            self.has_original = True
        return self._samples, self._labels
    
    def norm(self, samples: ndarray, is_cond: bool=False, epsilon: float=WHITEN_EPSILON) -> ndarray:
        if not self.has_preprocessed:
            print_msg("Whitening data have not yet been created for these " + \
                      "data. Run `preproc()` on this `DataManager` instance "+\
                      "before running `norm()`.",
                      level=LOG_ERROR)
            return None

        if is_cond:
            return whiten(samples, self.whiten_data, epsilon=epsilon, ret_dict=False)
        else:
            return whiten(samples, self.whiten_cond, epsilon=epsilon, ret_dict=False)
        
    def denorm(self, data: ndarray, is_cond: bool=False) -> ndarray:
        if not self.has_preprocessed:
            print_msg("Whitening data have not yet been created for these " + \
                      "data. Run `preproc()` on this `DataManager` instance "+\
                      "before running `denorm()`.",
                      level=LOG_ERROR)
            return None
        
        if is_cond:
            return dewhiten(data, self.whiten_data)
        else:
            return dewhiten(data, self.whiten_cond)
        
    def find(self, labels: ndarray) -> list[ndarray]:
        """
        Returns the samples correspdoning to the label in the data if they
        exist.
        """
        pass
        # TODO impl with intersect_labels


class ModelManager(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def compile_model(self) -> None:
        ...

    @abc.abstractmethod
    def train_model(self, data: ndarray, cond: ndarray, batch_size: int,
                    starting_epoch: int=0, end_epoch: int=1,
                    path: str|None=None, callbacks: list=None) -> None:
        ...

    @abc.abstractmethod
    def eval_model(self, cond) -> ndarray:
        ...

    @abc.abstractmethod
    def save_model(self) -> bool:
        ...