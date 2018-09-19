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

### Small Synthetic Dataset

This method uses RNNs, so requires time-series datasets. For initial testing,
some trivial datasets can be generated using `python3 datasets/generate_trivial_datasets.py`.
These trivial datasets are in the same format as those from
[UCR Time Series Classification Archive](http://www.cs.ucr.edu/~eamonn/time_series_data/).
You could also try downloading those UCR datasets and doing adaptation between
some of them; though, I have not tried this.

### MIMIC-III Health Care Dataset

The original paper used data from the
[MIMIC-III health care dataset](https://mimic.physionet.org/). We will use
[code provided](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII)
for [another paper](https://arxiv.org/abs/1710.08531) that creates various datasets from
MIMIC-III.

First you have to
[get access to MIMIC-III](https://mimic.physionet.org/gettingstarted/access/),
which requires taking an online CITI course.
After you're approved, then you can download MIMIC-III either following
[their examples](https://github.com/MIT-LCP/mimic-code) or by running
`datasets/download_mimic_iii.sh` after adding a line with your username and password
to your *~/.netrc* file:

    machine physionet.org login <yourusername> password <yourpassword>

Once downloaded (~6.6 GiB), you can generate the .tfrecord files by running
`python3 datasets/generate_mimic_iii.py`.

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
