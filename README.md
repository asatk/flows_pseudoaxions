<!--flows README-->
- [Current Bugs/Issues/"Features"](#current-bugsissuesfeatures)
- [Goals of Flows-Enriched Data Generation for High-Energy EXperiment (FEDHEX)](#goals-of-flows-enriched-data-generation-for-high-energy-experiment-fedhex)
- [Getting Started](#getting-started)
  - [Generating Data](#generating-data)
  - [Running An Experiment](#running-an-experiment)
  - [Performing Analysis](#performing-analysis)

# Current Bugs/Issues/"Features"
 - networks with multiple layers are not being saved or loaded correctly. They don't get evaluated correctly at the very least.
 - The checkpointing callback does not save exactly at the epoch interval desired. I think this is likely a rounding issue as it looks to save based on the n-th batch, not the n-th epoch. So dividing total data size by batchsize produces yields an extra training step in the epoch to finish the last batch/cover the remainder. The math for the saving frequency for ckpt does not take this into account. Perhaps find a way to align checkpointing with epochs, though this is more an aesthetic/meta-accuracy thing than critical to the model's success.

# Goals of Flows-Enriched Data Generation for High-Energy EXperiment (FEDHEX) 

Given a sparse collection of event distributions in an N-dimensional parameter space, we want to interpolate between the given distributions to generate new distributions

For example, a hypothetical interaction between two particles, yielding a scalar $\Phi$ and pseudoaxion (pseudoscalar) $\omega$, can be modelled as such.

![Sparse grid of distributions between which our framework can estimate an intermediate distribution by interpolating the features of nearby distributions.](readme_imgs/plotroot.png "10x10 Sparse Grid of Reconstructed Particle Masses")

We wish to get the accuracy to within ~1% of MCMC-generated data.

# Getting Started

Create a new environment using (ana/mini)conda package manager:

```conda create -n flows3.10 --file requirements.txt```

## Generating Data


## Running An Experiment


To run the network:


--- OR ---

``` py

```

to load a saved model:
``` py

```

to use a saved flow
``` py

```


Look how a model network performs:
![The training loss of the flow as it trains. Plotted is the loss on the y-axis in log-scale against the epoch at which it was recorded on the x-axis. When the losss becomes negative, its absolute value is plotted.](readme_imgs/loss.png "The training losses of a flow with 10 bijections, 1 layer and 128 parameters per bijection.")


## Performing Analysis


![Analysis of training and generated data for the same label. Four plots are shown: the top row showing each distribution individually, the bottom row showing them on the same plot with the same scale and their binned residuals.](readme_imgs/res.png "Comparison between training and generated data on label (Phi=2464, Omega=5.125)")