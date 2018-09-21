# Datasets

This method uses RNNs, so requires time-series datasets. I started with some
synthetic datasets, tried a RF sleep stage dataset, then also used the MIMIC-III
dataset that was used in the VRADA paper.

## Small Synthetic Dataset

For initial testing, some trivial datasets can be generated using
`python3 datasets/generate_trivial_datasets.py`.
These trivial datasets are in the same format as those from
[UCR Time Series Classification Archive](http://www.cs.ucr.edu/~eamonn/time_series_data/).
You could also try downloading those UCR datasets and doing adaptation between
some of them; though, I have not tried this.

## RF Sleep Stage Dataset

There's a paper about
[learning sleep stages](http://proceedings.mlr.press/v70/zhao17d.html) from
radio frequency (RF) data. I've also tried out domain adaptation on this dataset.
Though, their paper is more domain generalization than domain adaptation since
during training the target data (even unlabeled) is not seen at all. You can
request access to their dataset on [their website](http://sleep.csail.mit.edu/).
Then, put the *.npy* files into *datasets/RFSleep*

## MIMIC-III Health Care Dataset

The original paper used data from the
[MIMIC-III health care dataset](https://mimic.physionet.org/). We will use
[code provided](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII)
for [another paper](https://arxiv.org/abs/1710.08531) that creates various datasets from
MIMIC-III.

### Request Access and Download

First you have to
[get access to MIMIC-III](https://mimic.physionet.org/gettingstarted/access/),
which requires taking an online CITI course.
After you're approved, then you can download MIMIC-III either following
[their examples](https://github.com/MIT-LCP/mimic-code) or by running
`datasets/download_mimic_iii.sh` after adding a line with your username and password
to your *~/.netrc* file:

    machine physionet.org login <yourusername> password <yourpassword>

It's approximately 6.6 GiB.

### Setting up PostgreSQL

Next, you need to create a PostgreSQL database for the data. Make sure you have
at least 60 GiB of free space on your hard drive.

I'm on Arch, so I installed with
`sudo pacman -S postgresql`. Then since I don't have enough disk space on my root
partition, I told it to put the database in my home directory by running
`sudo -E systemctl edit postgresql.service` (replace "/pathto" with wherever
you want to put the database).

    [Service]
    Environment=PGROOT=/path/pgroot
    PIDFile=/pathto/pgroot/data/postmaster.pid
    ProtectHome=false

Created the directory, set to proper user, and then initialize database.

    pgroot="/pathto/pgroot"
    mkdir -p "$pgroot/data"
    sudo chown -R postgres:postgres "$pgroot"
    sudo -u postgres initdb -D "$pgroot/data"

Start the database.

    sudo systemctl daemon-reload
    sudo systemctl start postgresql

Create database and schema.

    sudo -u postgres createdb mimic
    psql -d mimic -c "CREATE SCHEMA mimiciii;" postgres

### Create the Database

Now we can create the database with the MIMIC-III scripts.

    datadir="../../../../mimic-iii/"
    cd datasets/mimic-code/buildmimic/postgres
    make mimic-gz datadir="$datadir"
    cd -

This likely will take many hours. It took 8 hours on my computer (not an SSD),
and it'll take up about 60 GiB.

### Process to Get Time-Series Data

You need to run all the processing to get time-series datasets out of the
database. First, make sure you have all the
[dependencies](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII) installed.

    sudo pacman -S python-psycopg2 python-numpy python-scipy python-scikit-learn \
        python-matplotlib python-pandas

Next, tell it how to connect to your database by editing
`getConnection()` in *datasets/process-mimic-iii/Codes/mimic3_mvcv/utils.py*

    def getConnection():
        return psycopg2.connect("dbname='mimic' user='postgres' password='postgres'")

Run a bunch of notebooks to generate the time-series data. Alternatively if you
prefer using a GUI rather than the terminal, you can open them as usual with
`jupyter-notebook`.

    cd datasets/process-mimic-iii/Codes/mimic3_mvcv
    mkdir -p res
    alias jnb="jupyter nbconvert --execute --ExecutePreprocessor.timeout=-1 --to notebook"

    jnb "0_createAdmissionList.ipynb"
    jnb "1_getItemIdList.ipynb"
    jnb "2_filterItemId_input.ipynb"
    jnb "3_filterItemId_output.ipynb"
    jnb "4_filterItemId_chart.ipynb"
    jnb "5_filterItemId_lab.ipynb"
    jnb "6_filterItemId_microbio.ipynb"
    jnb "7_filterItemId_prescript.ipynb"
    jnb "8_processing.ipynb"
    jnb "9_collect_mortality_labels.ipynb"
    jnb "9_getValidDataset.ipynb"
    jnb "10_get_17-features-processed(fromdb).ipynb"
    jnb "10_get_17-features-raw.ipynb"
    jnb "10_get_99plus-features-raw.ipynb"

Generate some more tables that are used during processing.

    alias psqlf='psql -d mimic postgres -f'

    psqlf "sql_gen_17features_ts/gen_gcs_ts.sql"
    psqlf "sql_gen_17features_ts/gen_lab_ts.sql"
    psqlf "sql_gen_17features_ts/gen_pao2_fio2.sql"
    psqlf "sql_gen_17features_ts/gen_urine_output_ts.sql"
    psqlf "sql_gen_17features_ts/gen_vital_ts.sql"
    psqlf "sql_gen_17features_ts/gen_17features_first24h.sql"
    psqlf "sql_gen_17features_ts/gen_17features_first48h.sql"

Run a few more notebooks.

    jnb "11_get_time_series_sample_17-features-processed_24hrs.ipynb"
    jnb "11_get_time_series_sample_17-features-processed_48hrs.ipynb"
    jnb "11_get_time_series_sample_17-features-raw_24hrs.ipynb"
    jnb "11_get_time_series_sample_17-features-raw_48hrs.ipynb"
    jnb "11_get_time_series_sample_99plus-features-raw_24hrs.ipynb"
    jnb "11_get_time_series_sample_99plus-features-raw_48hrs.ipynb"
    cd -

**Note:** because this process takes so long, I would highly recommend at least
doing this on an SSD. Though, I copied the PostgreSQL database to a high
performance cluster into RAM and ran all the processing there. This can be seen
in the Slurm script *../kamiak_process.srun*

### Create .tfrecord for TensorFlow

You can generate the .tfrecord files by running
`python3 datasets/generate_mimic_iii.py`.