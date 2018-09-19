# Variational Recurrent Adversarial Deep Domain Adaptation (VRADA)

Implementation of VRADA in TensorFlow. See
[their paper](https://openreview.net/pdf?id=rk9eAFcxg) or
[blog post](https://wcarvalho.github.io/research/2017/04/23/vrada/) for details
about the method. In their
[2016 workshop paper](https://pdfs.semanticscholar.org/eb6f/50e8a4dc7dafff1fb0dfa4046a46f0c1c3da.pdf),
they called this Variational Adversarial Deep Domain Adaptation (VADDA).  It's
more-or-less the same method though they might do iterative optimization
slightly differently.

You have a choice of running with or without domain adaptation and with two types
of RNNs. In their paper, they refer to the LSTM with domain adaptation as "R-DANN"
and the VRNN with domain adaptation as "VRADA."

 - *--lstm* -- use LSTM without adaptation
 - *--vrnn* -- use VRNN without adaptation
 - *--lstm-da* -- use LSTM with adaptation (R-DANN)
 - *--vrnn-da* -- use VRNN with adaptation (VRADA/VADDA)

To try these out, make sure you clone the repository recursively since there's submodules:

    git clone --recursive https://github.com/floft/vrada
    cd vrada

## Datasets

This method uses RNNs, so requires time-series datasets. See
[README.md in datasets/](https://github.com/floft/vrada/tree/master/datasets)
for information about generating some simple synthetic datasets or using the
MIMIC-III health care dataset that the VRADA paper used. (Note: it takes a few
days to get access to MIMIC-III.)

## Usage

### Training Locally

For example, to run domain adaption locally using a VRNN:

    python3 VRADA.py --logdir logs --modeldir models --imgdir images --debug --vrnn-da

### Training on a High-Performance Cluster

Alternatively, training on a cluster with
[Slurm](https://slurm.schedmd.com/overview.html)
(in my case on [Kamiak](https://hpc.wsu.edu/)) after editing
*kamiak_config.sh*:

    sbatch kamiak_train.sh --vrnn-da

Then on your local computer to monitor the progress in TensorBoard:

    ./kamiak_tflogs.sh
    tensorboard --logdir vrada-logs
