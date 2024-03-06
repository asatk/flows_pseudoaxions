import fedhex as fx
from matplotlib import pyplot as plt
from fedhex.train import Checkpointer, EpochLossHistory, SelectiveProgbarLogger
import numpy as np
from fedhex.posttrain import plot as fxp
import fedhex.io as fxio
import sys, getopt
import toml
import os

def printHelp():
    print('''usage: python fedHex.py CONFIG
                  
optional arguments:
  -h, --help                 show this message and exit
     
config option:   
  CONFIG                     Path to .TOML file used to define all network/running parameters. 
                             Can be overwritten by using options. See param.TOML for format.
   
execution options:   
  --plotTraining             Plot training data.
  -t True/False              Network will be trained if true. If false, network will
                             be loaded from flow_path.
  -e True/False              Network will evaluate to generate data if true.
   
run options:   
  --rootIn PATH              Path to directory containing root files for training.
  --rootThresh THRESHOLD     Threshold for training data.
  --nGen NUM                 Number of generated points.
  --labels LABEL1,LABEL2     Comma separarated label points i.e. 2464.0,5.125
  --rootOut PATH             Path to directory or .root file where the resulting root file will
                             be stored.
  --savePlot                 Save the plots that were created.
                  ''')

def main(argv):
    
    opts, args = getopt.getopt(argv[1:],"t:e:n:l:h",["rootIn=","rootThresh=", "nGen=", "rootOut=","labels=","plotTraining","help","savePlot"])
    
    if argv[0].endswith(".TOML") or argv[0].endswith(".toml"):
        configFile = argv[0]
        with open(configFile, "r") as f:
            param = toml.load(f)  
              
        #Modes
        train = param["modes"]["train"]
        evaluate = param["modes"]["evaluate"]
        plotTraining = param["modes"]["plotTraining"]
        
        #root loader
        root_path = param["rootIn"]["root_path"]
        thresh = param["rootIn"]["thresh"]
        tree_name = param["rootIn"]["tree_name"]
        data_vars = param["rootIn"]["data_vars"]
        cond_vars = param["rootIn"]["cond_vars"]
        cutstr = param["rootIn"]["cutstr"]
        defs = param["rootIn"]["defs"]
        
        #Generating
        ngen = param["generating"]["ngen"]
        gen_labels_unique = param["generating"]["gen_labels_unique"]
        ranges = param["generating"]["ranges"]
        for i in ranges:
            for j in range(len(i)):
                if i[j] == "inf":
                    i[j] = np.inf
                elif i[j] == "-inf":
                    i[j] = -np.inf
                else:
                    i[j] = float(i[j])

        #Outputs
        root_out_path = param["outputs"]["root_out_path"]
        loss_plot_path = param["outputs"]["loss_plot_path"]
        gen_plot_path = param["outputs"]["gen_plot_path"]
        train_plot_path = param["outputs"]["train_plot_path"]
        savePlot = param["outputs"]["savePlot"]
        
        #Network Hyperparameters
        nmade = param["network"]["compile"]["nmade"]
        ninputs = param["network"]["compile"]["ninputs"]
        ncinputs = param["network"]["compile"]["ncinputs"]
        hidden_layers = param["network"]["compile"]["hidden_layers"]
        hidden_units = param["network"]["compile"]["hidden_units"]
        lr_tuple = tuple(param["network"]["compile"]["lr_tuple"])
        #Training Paramaters
        starting_epoch = param["network"]["train"]["starting_epoch"]
        end_epoch = param["network"]["train"]["end_epoch"]
        batch_size = param["network"]["train"]["batch_size"]
        flow_path = param["network"]["train"]["flow_path"]
        loss_path = flow_path + "loss.npy"
        
        save_freq = param["callbacks"]["save_freq"] * batch_size
        log_freq = param["callbacks"]["log_freq"]
    elif argv[0] in ("-h", "--help"):
        printHelp()
        sys.exit()
    else:
        print("Please provide a .TOML config file as the first argument. Use -h or --help to learn more.")
        sys.exit()
    
    for opt, arg in opts:
        if opt == "-t":
            train = arg in ("True", "true", "T", 't')
        elif opt == "-e":
            evaluate = arg in ("True", "true", "T", "t")
        elif opt == "--rootIn":
            root_path = arg
        elif opt == "--rootThresh":
            thresh = int(arg)
        elif opt in ("--nGen", "-n"):
            ngen = int(arg)
        elif opt == "--rootOut":
            root_out_path = arg
        elif opt in ("--labels","-l"):
            gen_labels_unique = [arg.split(",")]
        elif opt == "--plotTraining":
            plotTraining = True
        elif opt == "--savePlot":
            savePlot = True
        elif opt in ("--help", "-h"):
         printHelp()
         sys.exit()


    ############################################################
    
    #load and preprocess root
    rl = fx.RootLoader(path=root_path)
    samples, labels = rl.load(
        tree_name=tree_name,
        data_vars=data_vars,
        cond_vars=cond_vars,
        cutstr=cutstr,
        defs=defs,
        event_thresh=thresh)
    data, cond = rl.preproc()
    
    #plot training data
    if plotTraining:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
        ax1.scatter(samples[:,0], samples[:,1])
        ax1.scatter(labels[:,0], labels[:,1])
        ax1.set_title("Original Data")
        ax1.set_xlabel(r"$\Phi$ reconstructed mass")
        ax1.set_ylabel(r"$\omega$ reconstructed mass")

        ax2.scatter(data[:,0], data[:,1])
        ax2.scatter(cond[:,0], cond[:,1])
        ax2.set_title("Preprocessed Data")
        ax2.set_xlabel("Normalized X-coordinate")
        ax2.set_ylabel("Normalized Y-coordinate")
        
        if savePlot:
            if not (train_plot_path.endswith(".png") or train_plot_path.endswith(".pdf") or train_plot_path.endswith(".jpg")):
                os.system("mkdir -p " + train_plot_path)
                train_plot_path += "training.png"
            fig.savefig(train_plot_path)
    

    if train:
        #compile model
        mm = fx.MADEManager(nmade=nmade, ninputs=ninputs, ncinputs=ncinputs,
                            hidden_layers=hidden_layers, hidden_units=hidden_units,
                            lr_tuple=lr_tuple)
        mm.compile_model()
    
        #callbacks
        callbacks = []
        
        os.system("mkdir -p " + flow_path)
        callbacks.append(Checkpointer(filepath=flow_path, save_freq=save_freq))

        callbacks.append(EpochLossHistory(loss_path=loss_path))

        callbacks.append(SelectiveProgbarLogger(1, epoch_interval=log_freq, epoch_end=end_epoch))
        #Train model
        mm.train_model(data=data, cond=cond, batch_size=batch_size,
                    starting_epoch=starting_epoch, end_epoch=end_epoch,
                    path=flow_path, callbacks=callbacks)
        
        #Save model
        mm.save(flow_path + "/config.json")
    else:
        #Load model
        if not os.path.isdir(flow_path):
            print("flow_path to model does not exist. Provide a path to an existing model to load or train new model.")
            sys.exit()
        mm = fx.MADEManager.import_model(path=flow_path)
        
    if evaluate:
        #Generate data
        gen_labels = np.repeat(gen_labels_unique, ngen, axis=0)
        gen_cond = rl.norm(gen_labels, is_cond=True)
        
        test, gen_data = mm.eval_model(gen_cond, rl, ranges=ranges)  
        gen_samples = gen_data

        #Plot data
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))

        ax1.scatter(gen_samples[:,0], gen_samples[:,1])
        ax1.scatter(gen_labels_unique[0][0], gen_labels_unique[0][1])
        ax1.set_title("Generated Data in Sample Space")
        ax1.set_xlabel(r"$\Phi$ reconstructed mass")
        ax1.set_ylabel(r"$\omega$ reconstructed mass")

        ax2.scatter(gen_data[:,0], gen_data[:,1])
        ax2.scatter(gen_cond[:,0], gen_cond[:,1])
        ax2.set_title("Generated Data")
        ax2.set_xlabel("Normalized X-coordinate")
        ax2.set_ylabel("Normalized Y-coordinate")
        
        if savePlot:
            if not (gen_plot_path.endswith(".png") or gen_plot_path.endswith(".pdf") or gen_plot_path.endswith(".jpg")):
                os.system("mkdir -p " + gen_plot_path)
                gen_plot_path += "generated.png"
            fig.savefig(gen_plot_path)
        
            if not (loss_plot_path.endswith(".png") or loss_plot_path.endswith(".pdf") or loss_plot_path.endswith(".jpg")):
                os.system("mkdir -p " + loss_plot_path)
                loss_plot_path += "loss.png"
            fxp.plot_losses(np.load(loss_path), out_path=loss_plot_path, show=True)
        else:
            fxp.plot_losses(np.load(loss_path), show=True)

        #Save data to root
        if not root_out_path.lower().endswith(".root"):
            os.system("mkdir -p " + root_out_path)
        fxio.save_root(
            path=root_out_path,
            tree_name="tree",
            gen_samples=gen_samples,
            gen_labels=gen_labels,
            trn_samples=samples,
            trn_labels=labels)
        


    
if __name__ == "__main__":
    main(sys.argv[1:])