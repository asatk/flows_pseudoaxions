"""
Author: Anthony Atkinson
Modified: 2023.07.15

Primary training script for running all procedures related to flows. This
includes data generation, model training, and analysis.
"""

# hopefully get rid of those stupid tensorflow warnings
from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import defs
import utils.analysis as autils
import utils.data as dutils
import utils.train as tutils
import utils.io as ioutils
from utils.train import MODE_LINE, MODE_GRID, MODE_ROOT

if __name__ == "__main__":

    # Perform initial checks of paths
    load_data_path = defs.root_dir
    training_data_path = defs.training_data_path
    config_path = defs.flow_path + "/config.json"

    mode = MODE_ROOT


    ioutils.init(output_dir=defs.output_dir, data_path=training_data_path,
                 flow_path=defs.flow_path, root_dir=defs.root_dir,
                 model_dir=defs.model_dir, newdata=defs.newdata,
                 newmodel=defs.newmodel, newanalysis=defs.newanalysis,
                 seed=defs.seed)
    ioutils.save_config(config_path, defs.__dict__)

    # Make new training data and save it
    if defs.newdata:

        # use-case-specific args
        args = {"event_threshold": defs.event_threshold}
        data, cond = dutils.makedata(mode, load_data_path=load_data_path,
                                     save_data_path=training_data_path,
                                     use_whiten=defs.normalize, overwrite=True,
                                     **args)

    # Load training data
    else:
        data, cond, _, _ = dutils.load_data_dict(training_data_path)

    
    model, distribution, made_list = tutils.getmodel(flow_path=defs.flow_path, newmodel=True)

    if defs.newmodel or defs.epoch_resume != 0:
        save_freq = int(defs.epoch_save * len(data) / defs.batch_size)
        tutils.train(model, data, cond, defs.nepochs, defs.batch_size,
                     flow_path=defs.flow_path, loss_path=defs.loss_log,
                     save_freq=save_freq)

    if defs.newanalysis:
        generated_data_path = None
        loss_log = defs.loss_log

        tools = [1, 3, 4, 5]
        lims = ((defs.phi_min, defs.phi_max), (defs.omega_min, defs.omega_max))

        autils.analyze(
            distribution, made_list, training_data_path, ngen=defs.ngen,
            lims=lims, generated_data_path=generated_data_path, tools=tools,
            loss_log=loss_log, output_dir=defs.output_dir,
            nworkers=defs.nworkers)
