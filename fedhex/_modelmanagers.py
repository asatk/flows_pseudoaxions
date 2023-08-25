from numpy import ndarray

from .io import save_config
from .train.tf import train
from .train.tf._MADEflow import compile_MADE_model, eval_MADE
from .utils import LOG_ERROR, print_msg

from ._managers import ModelManager


class MADEManager(ModelManager):
    """
    The details of building and training a model are self-contained within
    this class.
    """
    def __init__(self, nmade: int, ninputs: int, ncinputs: int,
                 hidden_layers: int, hidden_units: int,
                 lr_tuple: tuple[int]) -> None:
        
        super().__init__()
        self._is_compiled = False
        self._is_trained = False

        self._nmade = nmade
        self._ninputs = ninputs
        self._ncinputs = ncinputs
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units
        self._lr_tuple = lr_tuple

    def compile_model(self) -> None:
        """
        Compile a model with all of the necessary parameters. This Manager will
        keep references to instances of tf.Model, tfd.TransformedDistribution,
        and a list of MADE blocks, all used internally.
        """
        model, dist, made_list = compile_MADE_model(num_made=self._nmade,
            num_inputs=self._ninputs, num_cond_inputs=self._ncinputs,
            hidden_layers=self._hidden_layers, hidden_units=self._hidden_units,
            lr_tuple=self._lr_tuple)
        
        self._model = model
        self._dist = dist
        self._made_list = made_list
        self._is_compiled = True

    def train_model(self, data: ndarray, cond: ndarray, batch_size: int,
                    starting_epoch: int=0, end_epoch: int=1, 
                    path: str|None=None, callbacks: list=None) -> None:
        """
        Train the model once built.
        """

        if self._is_compiled is False:
            print_msg("The model is not compiled. Please use the instance " + \
                      "method `MADEManager.compile_model()` in order for " + \
                      "this model to be trainable.", level=LOG_ERROR)
            return

        if callbacks == None:
            callbacks = []

        self._nepochs = end_epoch
        self._batch_size = batch_size
        self._starting_epoch = starting_epoch
        self._flow_path = path
        
        train(self._model, data, cond, end_epoch=end_epoch, batch_size=batch_size,
              starting_epoch=starting_epoch, flow_path=path,
              callbacks=callbacks)
        self._is_trained = True
        
    def eval_model(self, cond) -> ndarray:

        if self._is_trained is False:
            print_msg("The model is not train. Please use the instance " + \
                      "method `MADEManager.train_model()` in order for " + \
                      "this model to be evaluatable.", level=LOG_ERROR)
            return None
        
        return eval_MADE(cond, self._made_list, self._dist)
    
    def save_model(self, flow_path: str) -> None:
        self._model.save(flow_path)

    def save(self, config_path: str) -> None:
        d = {"nmade": self._nmade, "ninputs": self._ninputs,
             "ncinputs": self._ncinputs, "hidden_layers": self._hidden_layers,
             "hidden_units": self._hidden_units, "lr_tuple": self._lr_tuple,
             "nepochs": self._nepochs, "batch_size": self._batch_size,
             "starting_epoch": self._starting_epoch,
             "flow_path": self._flow_path}
        save_config(config_path, d, save_all=False)


class RNVPManager(ModelManager):
    """
    Real Non-Volume Preserving flows are not implemented yet.
    """
    pass
