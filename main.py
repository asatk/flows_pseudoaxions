"""
Author: Anthony Atkinson
Modified: 2023.07.14

Primary training script for running all procedures related to flows. This
includes data generation, model training, and analysis.
"""

import defs
import utils.analysis as autils
import utils.data as dutils
import utils.train as tutils
import utils.io as ioutils

if __name__ == "__main__":

    # Perform initial checks of paths
    load_data_path = defs.root_dir
    training_data_path = defs.training_data_path
    config_path = defs.flow_path + "/config.json"


    ioutils.init(output_dir=defs.output_dir, data_path=training_data_path,
                 flow_path=defs.flow_path)
    ioutils.save_config(config_path, defs.__dict__)

    # Make new training data and save it
    if defs.newdata:
        data, cond = dutils.makedata(defs.mode, load_data_path=load_data_path,
                                     save_data_path=training_data_path,
                                     use_whiten=defs.normalize, overwrite=True)

    # Load training data
    else:
        data, cond, _, _ = dutils.load_data_dict(training_data_path)

    model, distribution, made_list = tutils.getmodel(defs.flow_path)

    if defs.newmodel or defs.epoch_resume != 0:
        tutils.train(model, data, cond, defs.nepochs, defs.batch_size,
                     flow_path=defs.flow_path, loss_path=defs.loss_log)

    if defs.newanalysis:
        generated_data_path = None
        loss_log = defs.loss_log

        tools = [1, 2, 3, 4, 5]

        autils.analyze(distribution, made_list, training_data_path,
                       generated_data_path=generated_data_path, tools=tools,
                       loss_log=loss_log, output_dir=defs.output_dir)
