instructions for use of network coming shortly (/!\under construction/!\)

pre-reqs for use:

python 3.10

create a new environment in (mini)conda:

conda create -n flows3.10 python=3.10 matplotlib numpy scikit-learn scipy \
        tensorflow-base tensorfow-probability uproot

necessary

matplotlib
multiprocessing (usually python builtin)
numpy
scikit-learn
scipy
tensorflow
tensorflow-probability
uproot

optional
bokeh (interactive visualization)
jupyter (good to test short bits of code)
pytorch (may port network to pytorch)
seaborn (good for visualization of scientific data)

to run the network:
python main.py

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