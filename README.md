<!--flows README-->
- [Getting Started](#getting-started)
  - [Running An Experiment](#running-an-experiment)
  - [Performing Analysis](#performing-analysis)

# Getting Started

Create a new environment using (ana/mini)conda package manager:

```conda create -n flows3.10 python=3.10 matplotlib numpy scikit-learn scipy tensorflow-base tensorflow-probability uproot```

| req. pkgs              | vsn    | opt. pkgs | vsn    | use                                       |
| :--------------------- | :----- | :-------- | :----- | :---------------------------------------- |
| matplotlib             | 3.7.1  | bokeh     | 3.1.1  | interactive visualization                 |
| numpy                  | 1.25.0 | jupyter   | 1.0.0  | good to test short bits of code           |
| python                 | 3.10   | pytorch   | 2.0.0  | may port network to pytorch               |
| scikit-learn           | 1.2.2  | seaborn   | 0.12.2 | good for visualization of scientific data |
| scipy                  | 1.10.1 |           |        |                                           |
| tensorflow             | 2.11.1 |           |        |                                           |
| tensorflow-probability | 0.19.0 |           |        |                                           |
| uproot                 | 5.0.8  |           |        |                                           |

## Running An Experiment

To run the network:
```python main.py```

--- OR ---

``` py
model.fit(x=[samples, labels],
            y=np.zeros((samples.shape[0], 0), dtype=np.float32),
            shuffle=True,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            initial_epoch=defs.epoch_resume if retrain else 0,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    filepath=ckptpath,
                    verbose=1,
                    save_weights_only=False,
                    save_freq=int(defs.epoch_save * len(samples) / batch_size)),
                myutils.SelectiveProgbarLogger(
                    verbose=1,
                    epoch_interval=10,
                    epoch_end=epochs)])
```

to load a saved model:
``` py
model = keras.models.load_model(modeldir, custom_objects={
        "lossfn": flowmodel.lossfn, "Made": flowmodel.Made})

made_blocks = []
for i in range(nmade):
    made_blocks.append(model.get_layer(name=f"made_{i}"))
    
distribution, made_list = flowmodel.build_distribution(made_blocks, ndim)
```

to use a saved flow
``` py
current_kwargs = {}
gen_labels = np.repeat(testlabels, nsamples, axis=0)
for i in range(nmade):
    current_kwargs[f"maf_{i}"] = {"conditional_input" : gen_labels}

gen_data = np.array(distribution.sample((nsamples, ),    
        bijector_kwargs=current_kwargs))
```

Important locations (these are defaults that can be changed):

Training data are stored in ```./data``` in the numpy .npy format

Models are stored ```./model``` in the keras SavedModel format from which they can be loaded in their entirety once constructing the modeul using ```flowmodel.build_distribution```.

Output from analysis is stored in ```./output```

When loading ROOT data, the loading algorithm traverses the entire directory tree looking for .ROOT files. Make sure there are ONLY directories or .ROOT files in the entire directory tree specified by ```root_dir```

## Performing Analysis

