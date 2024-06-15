import fedhex as fx
from fedhex.train import Checkpointer, EpochLossHistory, SelectiveProgbarLogger
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

def plot_data(samples, labels):
    plt.scatter(samples[:,0], samples[:,1])
    plt.scatter(labels[:,0], labels[:,1])
    plt.savefig("output/data-scatter.png")

def plot_gen_grouped(gen_samples_grouped, gen_labels_unique):
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2,4, figsize=(12,8), sharex=True, sharey=True)
    ax1: plt.Axes
    ax2: plt.Axes
    ax3: plt.Axes
    ax4: plt.Axes
    ax5: plt.Axes
    ax6: plt.Axes
    ax7: plt.Axes
    ax8: plt.Axes

    ax1.scatter(gen_samples_grouped[0][:,0], gen_samples_grouped[0][:,1])
    ax1.scatter(gen_labels_unique[0,0], gen_labels_unique[0,1], label=gen_labels_unique[0])

    ax2.scatter(gen_samples_grouped[1][:,0], gen_samples_grouped[1][:,1])
    ax2.scatter(gen_labels_unique[1,0], gen_labels_unique[1,1], label=gen_labels_unique[1])

    ax3.scatter(gen_samples_grouped[2][:,0], gen_samples_grouped[2][:,1])
    ax3.scatter(gen_labels_unique[2,0], gen_labels_unique[2,1], label=gen_labels_unique[2])

    ax4.scatter(gen_samples_grouped[3][:,0], gen_samples_grouped[3][:,1])
    ax4.scatter(gen_labels_unique[3,0], gen_labels_unique[3,1], label=gen_labels_unique[3])

    ax5.scatter(gen_samples_grouped[4][:,0], gen_samples_grouped[4][:,1])
    ax5.scatter(gen_labels_unique[4,0], gen_labels_unique[4,1], label=gen_labels_unique[4])

    ax6.scatter(gen_samples_grouped[5][:,0], gen_samples_grouped[5][:,1])
    ax6.scatter(gen_labels_unique[5,0], gen_labels_unique[5,1], label=gen_labels_unique[5])

    ax7.scatter(gen_samples_grouped[6][:,0], gen_samples_grouped[6][:,1])
    ax7.scatter(gen_labels_unique[6,0], gen_labels_unique[6,1], label=gen_labels_unique[6])

    ax8.scatter(gen_samples_grouped[7][:,0], gen_samples_grouped[7][:,1])
    ax8.scatter(gen_labels_unique[7,0], gen_labels_unique[7,1], label=gen_labels_unique[7])

    ax1.set_xlim((0, 3000))
    ax1.set_ylim((0, 12))

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    ax7.legend()
    ax8.legend()

    fig.suptitle("Generated Data (Phi = 500GeV)")
    fig.supxlabel("Reco Phi Mass")
    fig.supylabel("Reco Omega Mass")
    fig.tight_layout()

    plt.savefig("output/gen-scatter-panels.png")

if __name__ == "__main__":

    if not os.path.isdir("./output/"):
        os.mkdir("./output/")
    if not os.path.isdir("./model/"):
        os.mkdir("./model/")
    
    path = sys.argv[1]
    print("Path: ", path)
#    path = "./root/100x100box_200events"
    tree_name = "Events"
    data_vars = ["phi", "omega"]
    cond_vars = ["labelphi", "labelomega"]
    cutstr = "CBL_Region == 1 and Photon_pt[CBL_RecoPhi_photonindex] > 220"
    defs = {
        "phi": "CBL_RecoPhi_mass", 
        "labelphi": "GenPhi_mass[0]",
        "omega": "TwoProng_massPi0[CBL_RecoPhi_twoprongindex]",
        "labelomega": "GenOmega_mass[0]"
    }

    rl = fx.RootLoader(path=path)
    samples, labels = rl.load(tree_name=tree_name,
                        data_vars=data_vars,
                        cond_vars=cond_vars,
                        cutstr=cutstr,
                        defs=defs,
                        event_thresh=0)
    data, cond = rl.preproc()
    plot_data(samples, labels)



    nmade = 10
    ninputs = 2
    ncinputs = 2
    hidden_layers = [128]
    lr_tuple = (1e-3, 1e-4, 100)
    mm = fx.MAFManager(num_flows=nmade,
                len_event=ninputs,
                len_cond_event=ncinputs,
                hidden_layers=hidden_layers,
                activation="relu",
                lr_tuple=lr_tuple)
    mm.compile_model()



    batch_size = int(np.power(2, np.floor(np.log2(len(data) >> 5))))
    model_path = "./model/2024-04-04-job/"
    loss_path = model_path + "/loss.npy"
    starting_epoch = 0
    end_epoch = 200

    callbacks = []

    save_freq = 50 * batch_size
    callbacks.append(Checkpointer(filepath=model_path, save_freq=save_freq))

    callbacks.append(EpochLossHistory(loss_path=loss_path))

    log_freq = 10
    callbacks.append(SelectiveProgbarLogger(1, epoch_interval=log_freq, epoch_end=end_epoch))



    mm.train_model(data=data,
            cond=cond,
            batch_size=batch_size,
            initial_epoch=starting_epoch,
            epochs=end_epoch,
            path=model_path,
            callbacks=callbacks)
    config_path = model_path + "/config.json"
    mm.save(config_path=config_path)
    mm = fx.MAFManager.import_model(path="./model/2024-04-04-job/")



    ngen = 500
    gen_labels_unique = np.array([[125, 1.0], [125, 2.0], [125, 5.0], [125, 9.9], [500, 1.0], [500, 2.0], [500, 5.0], [500, 9.9]])
    gen_labels = np.repeat(gen_labels_unique, ngen, axis=0)
    # gen_cond = rl.norm(gen_labels, is_cond=True)

    gen_samples = mm.eval_model(gen_labels, rl, criteria=lambda x: x[:,1] < 0.415)

    unique_labels, unique_inverse = np.unique(gen_labels, return_inverse=True, axis=0)
    gen_samples_grouped = [gen_samples[unique_inverse == i] for i in range(len(gen_labels_unique))]



    plot_gen_grouped(gen_samples_grouped, gen_labels_unique)
