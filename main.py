"""
Author: Anthony Atkinson
Modified: 2023.07.15

Primary training script for running all procedures related to flows. This
includes data generation, model training, and analysis.
"""

# hopefully get rid of those stupid tensorflow warnings
from absl import logging as absl_logging
from os import environ
absl_logging.set_verbosity(absl_logging.ERROR)
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import defs
from fedhex.train.tf import _train as tutils
from fedhex.io import _path as ioutils

if __name__ == "__main__":

    # Perform initial checks of paths
    load_data_path = defs.root_dir
    training_data_path = defs.training_data_path
    config_path = defs.flow_path + "/config.json"

    mode = MODE_ROOT

    defs_dict = defs.__dict__


    # Make new training data and save it
    if defs.newdata:

        config_data_path = defs.data_dir + defs.data_name + "/config_d.json"
        config_data_kws = [""]
        config_data_dict = {k: v for k, v in defs_dict.items() if k in config_data_kws}
        ioutils.save_config(config_data_path, config_data_dict)

        dutils.init()

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

        config_model_path = defs.flow_path + "/config_m.json"
        config_model_kws = [""]
        config_model_dict = {}
        ioutils.save_config(config_model_path, config_model_dict)

        save_freq = int(defs.epoch_save * len(data) / defs.batch_size)
        tutils.train(model, data, cond, defs.nepochs, defs.batch_size,
                     flow_path=defs.flow_path, loss_path=defs.loss_log,
                     save_freq=save_freq)

    if defs.newanalysis:

        generated_data_path = None
        loss_log = defs.loss_log

        # tools = [1, 3, 4, 5]
        tools = [5]
        lims = ((defs.phi_min, defs.phi_max), (defs.omega_min, defs.omega_max))

        
        config_analysis_path = defs.output_dir + "config_a.json"
        config_analysis_dict = {}
        ioutils.save_config(config_analysis_path, config_analysis_dict)
        
        autils.analyze(
            distribution, made_list, training_data_path, ngen=defs.ngen,
            lims=lims, generated_data_path=generated_data_path, tools=tools,
            loss_log=loss_log, output_dir=defs.output_dir,
            nworkers=defs.nworkers)
