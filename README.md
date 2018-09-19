# Variational Recurrent Adversarial Deep Domain Adaptation (VRADA)

Implementation of VRADA in TensorFlow. See
[their paper](https://openreview.net/pdf?id=rk9eAFcxg) or
[blog post](https://wcarvalho.github.io/research/2017/04/23/vrada/) for details
about the method. In their
[2016 workshop paper](https://pdfs.semanticscholar.org/eb6f/50e8a4dc7dafff1fb0dfa4046a46f0c1c3da.pdf),
they called this Variational Adversarial Deep Domain Adaptation (VADDA).  It's
more-or-less the same method though they might do iterative optimization
slightly differently.

## Datasets

This method uses RNNs, so requires time-series datasets. Some trivial datasets
can be generated using `python3 datasets/generate_trivial_datasets.py`. These
trivial datasets are in the same format as those from
[UCR Time Series Classification Archive](http://www.cs.ucr.edu/~eamonn/time_series_data/).

The original paper used data from the
[MIMIC-III health care dataset](https://mimic.physionet.org/). This will be
added shortly.

## Usage

### Training Locally

For example, to run domain adaption locally using a VRNN:

    python3 VRADA.py --logdir logs --modeldir models \
        --imgdir images --debug --vrnn-da

### Training on a High-Performance Cluster

Alternatively, training on a cluster with
[Slurm](https://slurm.schedmd.com/overview.html)
(in my case on [Kamiak](https://hpc.wsu.edu/)) after editing
*kamiak_config.sh*:

    sbatch kamiak_train.sh --vrnn-da

Then on your local computer to monitor the progress in TensorBoard:

    ./kamiak_tflogs.sh
    tensorboard --logdir vrada-logs
