import os
import shutil
from typing import Callable

from numpy import ceil, log2, ndarray
import tensorflow as tf
from tensorflow import keras
from tensorflow_probability.python.distributions import Distribution

from ._managers import DataManager, ModelManager
from .train.tf import train, NLL
from .train.tf._MAF import compile_MAF, eval_MAF, load_MAF
from .utils import LOG_ERROR, print_msg


class MAFManager(ModelManager):
    """
    A Tensorflow-based Masked Autoregressive Flow model can be built, compiled,
    trained, evaluated, and exported using this ModelManager.
    """
    def __init__(self,
                 num_flows: int,
                 len_event: int,
                 len_cond_event: int,
                 hidden_units: list[int],
                 activation: str="relu"
                 ) -> None:
        """Create a manager for a Masked Autoregressive Flow/chain of MADE
        bijections.

        Args:
            num_flows (int): _description_
            len_event (int): _description_
            len_cond_event (int): _description_
            hidden_units (list[int]): _description_
            activation (str, optional): _description_. Defaults to "relu".
        """        
        
        super().__init__()

        self._num_flows = num_flows
        self._len_event = len_event
        self._len_cond_event = len_cond_event
        self._hidden_units = hidden_units
        self._activation = activation

        self.state_dict.update({
            "num_flows": num_flows,
            "len_event": len_event,
            "len_cond_event": len_cond_event,
            "hidden_units": hidden_units,
            "activation": activation
        })


    @classmethod
    def import_model(cls,
                     path: str,
                     loss: Callable|keras.losses.Loss=NLL):

        model, dist, made_list, cfg = load_MAF(flow_path=path, loss=loss)
        mm = MAFManager(num_flows=cfg["num_flows"],
                         len_event=cfg["len_event"],
                         len_cond_event=cfg["len_cond_event"],
                         hidden_units=cfg["hidden_units"],
                         activation=cfg["activation"])
        
        mm._model = model
        mm._dist = dist
        mm._made_list = made_list
        mm._opt = model.optimizer
        mm.is_compiled = True
        mm.is_trained = True

        return mm


    def compile_model(self,
                      prior: Distribution=None,
                      optimizer: keras.optimizers.Optimizer=None,
                      loss: Callable[[float, float], float]|tf.losses.Loss=None) -> None:
        """
        Compile a model with all of the necessary parameters. This Manager will
        keep references to instances of tf.Model, tfd.TransformedDistribution,
        and a list of MADE blocks, all used internally.
        """
        model, dist, made_list = compile_MAF(
            num_flows=self._num_flows,
            len_event=self._len_event,
            len_cond_event=self._len_cond_event,
            hidden_units=self._hidden_units,
            activation=self._activation,
            prior=prior,
            optimizer=optimizer,
            loss=loss)
        
        self._prior = prior
        self._opt = optimizer
        self._loss = loss
        self._model = model
        self._dist = dist
        self._made_list = made_list
        self.is_compiled = True


    def train_model(self,
                    dm: DataManager=None,
                    data: ndarray=None,
                    cond: ndarray=None,
                    batch_size: int=None,
                    initial_epoch: int=0,
                    epochs: int=1, 
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

        self._end_epoch = epochs
        self._batch_size = batch_size
        self._starting_epoch = initial_epoch
        self._model_path = path

        self.state_dict.update({
            "end_epoch": epochs,
            "batch_size": batch_size,
            "starting_epoch": initial_epoch,
            "model_path": path
        })

        if path is not None:
            if os.path.isdir(path):
                shutil.rmtree(path)
            os.mkdir(path)
        
        train(self._model,
              data,
              cond,
              epochs=epochs,
              batch_size=batch_size,
              initial_epoch=initial_epoch,
              flow_path=path,
              callbacks=callbacks)
        self.is_trained = True
        
    def eval_model(self,
                   cond: ndarray,
                   dm: DataManager=None,
                   criteria: Callable=None,
                   ranges: list[list[float]]=None,
                   seed: int=0x2024,
                   *args) -> ndarray:

        if self.is_trained is False:
            print_msg("The model is not trained. Please use the instance " + \
                      "method `MADEManager.train_model()` in order to " + \
                      "evaluate this model.", level=LOG_ERROR)
            return None
        
        return eval_MAF(cond,
                         self._made_list,
                         self._dist,
                         dm=dm,
                         criteria=criteria,
                         ranges=ranges,
                         seed=seed,
                         *args)
    

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
