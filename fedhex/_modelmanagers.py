from numpy import ndarray

import os
import shutil
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
    def __init__(self,
                 nmade: int,
                 ninputs: int,
                 ncinputs: int,
                 hidden_layers: int|list=1,
                 hidden_units: int=128,
                 activation: str="relu",
                 lr_tuple: tuple[int]=(1e-3, 1e-4, 100)) -> None:
        
        super().__init__()

        self._nmade = nmade
        self._ninputs = ninputs
        self._ncinputs = ncinputs
        self._hidden_layers = hidden_layers
        self._hidden_units = hidden_units
        self._activation = activation
        self._lr_tuple = lr_tuple

        self.state_dict.update({
            "nmade": nmade,
            "ninputs": ninputs,
            "ncinputs": ncinputs,
            "hidden_layers": hidden_layers,
            "hidden_units": hidden_units,
            "activation": activation,
            "lr_tuple": lr_tuple
        })

    def compile_model(self) -> None:
        """
        Compile a model with all of the necessary parameters. This Manager will
        keep references to instances of tf.Model, tfd.TransformedDistribution,
        and a list of MADE blocks, all used internally.
        """
        model, dist, made_list = compile_MADE_model(num_made=self._nmade,
            num_inputs=self._ninputs, num_cond_inputs=self._ncinputs,
            hidden_layers=self._hidden_layers, hidden_units=self._hidden_units,
            activation=self._activation, lr_tuple=self._lr_tuple)
        
        self._model = model
        self._dist = dist
        self._made_list = made_list
        self.is_compiled = True

    def train_model(self,
                    data: ndarray,
                    cond: ndarray,
                    batch_size: int,
                    starting_epoch: int=0,
                    end_epoch: int=1, 
                    path: str|None=None,
                    callbacks: list=None) -> None:
        """
        Train the model once built.
        """

        if self.is_compiled is False:
            print_msg("The model is not compiled. Please use the instance " + \
                      "method `MADEManager.compile_model()` in order to " + \
                      "train this model.", level=LOG_ERROR)
            return

        if callbacks == None:
            callbacks = []

        self._end_epoch = end_epoch
        self._batch_size = batch_size
        self._starting_epoch = starting_epoch
        self._model_path = path

        self.state_dict.update({
            "end_epoch": end_epoch,
            "batch_size": batch_size,
            "starting_epoch": starting_epoch,
            "model_path": path
        })

        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)
        
        train(self._model, data, cond, end_epoch=end_epoch, batch_size=batch_size,
              starting_epoch=starting_epoch, flow_path=path,
              callbacks=callbacks)
        self.is_trained = True
        
    def eval_model(self, cond) -> ndarray:

        if self.is_trained is False:
            print_msg("The model is not trained. Please use the instance " + \
                      "method `MADEManager.train_model()` in order to " + \
                      "evaluate this model.", level=LOG_ERROR)
            return None
        
        return eval_MADE(cond, self._made_list, self._dist)
    
    def export_model(self, path: str) -> bool:
        if not self.is_compiled:
            print_msg("This model is not compiled. Please use the instance" + \
                      "method `compile_model()` in order to save this model.",
                      level=LOG_ERROR)
            return False
        
        self._model.save(path)
        return True


class RNVPManager(ModelManager):
    """
    Real Non-Volume Preserving flows are not implemented yet.
    """
    def __init__(self):
        super().__init__()
