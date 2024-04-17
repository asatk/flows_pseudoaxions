from numpy import ndarray, ceil, log2

import os
import shutil
from .io import save_config
from .train.tf import train
from .train.tf._MADEflow import compile_MADE_model, eval_MADE, load_MADE
from .utils import LOG_ERROR, print_msg

from ._managers import ModelManager, DataManager


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

    @classmethod
    def import_model(cls, path: str):

        model, dist, made_list, cfg = load_MADE(flow_path=path, newmodel=False)
        mm = MADEManager(nmade=cfg["nmade"],
                         ninputs=cfg["ninputs"],
                         ncinputs=cfg["ncinputs"],
                         hidden_layers=cfg["hidden_layers"],
                         hidden_units=cfg["hidden_units"],
                         activation=cfg["activation"],
                         lr_tuple=cfg["lr_tuple"])
        
        mm._model = model
        mm._dist = dist
        mm._made_list = made_list
        mm.is_compiled = True
        mm.is_trained = True

        return mm

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
                    dm: DataManager=None,
                    data: ndarray=None,
                    cond: ndarray=None,
                    batch_size: int=None,
                    starting_epoch: int=0,
                    end_epoch: int=1, 
                    path: str|None=None,
                    callbacks: list=None) -> None:
        """
        Train the model once built.

        dm preferred over data/cond
        """

        if self.is_compiled is False:
            print_msg("The model is not compiled. Please use the instance " + \
                      "method `MADEManager.compile_model()` in order to " + \
                      "train this model.", level=LOG_ERROR)
            return

        if dm is not None:
            data, cond = dm.preproc()
        elif data is None or cond is None:
            print_msg("The model cannot be trained without data and " + \
                      "conditional data or just a DataManager. Please provide"+\
                      "either of these.")

        if batch_size is None:
            batch_size = int(1 << ceil(log2(len(data) >> 5)))
            print_msg("Note: no batch_size provided; batch_size set to " + \
                      f"{batch_size}")

        if callbacks is None:
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

        if path is not None:
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.mkdir(path)
        
        train(self._model, data, cond, end_epoch=end_epoch, batch_size=batch_size,
              starting_epoch=starting_epoch, flow_path=path,
              callbacks=callbacks)
        self.is_trained = True
        
    def eval_model(self, cond, dm: DataManager, criteria = None, ranges=None, seed: int=0x2024, *args) -> ndarray:

        if self.is_trained is False:
            print_msg("The model is not trained. Please use the instance " + \
                      "method `MADEManager.train_model()` in order to " + \
                      "evaluate this model.", level=LOG_ERROR)
            return None
        
        return eval_MADE(cond, dm, self._made_list, self._dist, criteria, ranges, seed=seed, *args)
    
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
