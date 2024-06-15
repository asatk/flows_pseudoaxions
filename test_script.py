import fedhex as fx
from fedhex.pretrain import generation as fxgen
from fedhex.posttrain import plot as fxp
from fedhex.train import Checkpointer, EpochLossHistory, SelectiveProgbarLogger
from matplotlib import pyplot as plt
import numpy as np


def generate_gaussian_data(gaus_spec: tuple[int, int], plot: bool=True):
    # List data generation parameters
    ndim = 2
    sigma = 0.025
    ngausx = gaus_spec[0]
    ngausy = gaus_spec[1]
    lims = ((0., 1.), (0., 1.))
    seed = 0xace1ace1ace1ace1

    # Specify generation 'Strategy' for this (gaussian) Generator
    # Strategies for gaussian Generators specify the behavior of
    # a base covariance matrix that changes over the sample space
    strat = fxgen.RepeatStrategy(fxgen.DiagCov(ndim=ndim, sigma=sigma))

    # Make a data generator instance for the specified gaussians
    ggg = fx.GridGaussGenerator(cov_strat=strat, ngausx=ngausx,
                                ngausy=ngausy, seed=seed, lims=lims)
    nsamp = 1000
    samples, labels = ggg.generate(nsamp=nsamp)
    data, cond = ggg.preproc()


    if plot:
        _, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
        ax1: plt.Axes
        ax2: plt.Axes

        ax1.scatter(samples[:,0], samples[:,1])
        ax1.scatter(labels[:,0], labels[:,1])
        ax1.set_title("Original Samples")
        ax1.set_xlabel("X-coordinate")
        ax1.set_ylabel("Y-coordinate")

        ax2.scatter(data[:,0], data[:,1])
        ax2.scatter(cond[:,0], cond[:,1])
        ax2.set_title("Preprocessed Data")
        ax2.set_xlabel("Normalized X-coordinate")
        ax2.set_ylabel("Normalized Y-coordinate")

        plt.show()
        plt.close()

    return samples, labels, data, cond, ggg

def make_model(id: str, end_epoch: int, batch_size: int):
    nmade = 10
    ninputs = 2
    ncinputs = 2
    hidden_layers = 1
    hidden_units = 128
    lr_tuple = (1e-3, 1e-4, 100)
    mm = fx.MAFManager(num_flows=nmade, len_event=ninputs, len_cond_event=ncinputs,
                        hidden_layers=hidden_layers, hidden_units=hidden_units,
                        lr_tuple=lr_tuple)
    
    mm.compile_model()
    
    flow_path = f"./model/11-16_{id}/"
    loss_path = flow_path + "loss.npy"

    callbacks = []
    save_freq = 50 * batch_size
    callbacks.append(Checkpointer(filepath=flow_path, save_freq=save_freq))
    callbacks.append(EpochLossHistory(loss_path=loss_path))
    log_freq = 10
    callbacks.append(SelectiveProgbarLogger(1, epoch_interval=log_freq, epoch_end=end_epoch))

    return mm, callbacks, flow_path


def test_model(label: tuple[float, float], output_path: str, ggg: fx.DataManager, mm: fx.ModelManager, plot: bool=False):
    ngen = 500
    gen_labels_unique = label
    gen_labels = np.repeat([gen_labels_unique], ngen, axis=0)
    gen_cond = ggg.norm(gen_labels, is_cond=True)
    gen_data = mm.eval_model(gen_cond)
    gen_samples = ggg.denorm(gen_data, is_cond=False)

    _, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    ax1: plt.Axes
    ax2: plt.Axes

    ax1.scatter(gen_samples[:,0], gen_samples[:,1])
    ax1.scatter(gen_labels_unique[0], gen_labels_unique[1])
    ax1.set_title("Generated Data in Sample Space")
    ax1.set_xlabel("X-coordinate")
    ax1.set_ylabel("X-coordinate")

    ax2.scatter(gen_data[:,0], gen_data[:,1])
    ax2.scatter(gen_cond[:,0], gen_cond[:,1])
    ax2.set_title("Generated Data in Normalized Data Space")
    ax2.set_xlabel("Normalized X-coordinate")
    ax2.set_ylabel("Normalized Y-coordinate")

    plt.savefig(output_path)

    if plot:
        plt.show()

    plt.close()


if __name__ == "__main__":

    gaus_specs = [(3, 3), (5, 5), (8, 8), (10, 10), (12, 12), (15, 15)]
    test_labels = [[0.5, 0.5], [0.25, 0.5], [0.25, 0.25], [0.15, 0.25], [0.15, 0.15],
              [0.1, 0.1], [0.05, 0.05], [0.01, 0.01]]

    starting_epoch = 0
    end_epoch = 200
    batch_size = 1024

    for i, spec in enumerate(gaus_specs):
        samples, labels, data, cond, ggg = generate_gaussian_data(spec, plot=False)
        mm, callbacks, flow_path = make_model(id=str(i), end_epoch=end_epoch, batch_size=batch_size)

        mm.train_model(data=data, cond=cond, batch_size=batch_size,
               initial_epoch=starting_epoch, epochs=end_epoch,
               path=flow_path, callbacks=callbacks)
        
        for test_label in test_labels:
            plot_output_path = f"output/model_{str(i)}_label_{test_label[0]:.2f}_{test_label[1]:.2f}.png"
            print(f"Testing model {i} at label {str(test_label)}")
            print(f"Output plot to: {plot_output_path}")
            test_model(test_label, plot_output_path, ggg, mm, plot=False)
        




    