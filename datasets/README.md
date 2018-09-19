# Datasets

This method uses RNNs, so requires time-series datasets. I started with some
synthetic datasets and then also used the MIMIC-III dataset that was used in the
VRADA paper.

## Small Synthetic Dataset

For initial testing, some trivial datasets can be generated using
`python3 datasets/generate_trivial_datasets.py`.
These trivial datasets are in the same format as those from
[UCR Time Series Classification Archive](http://www.cs.ucr.edu/~eamonn/time_series_data/).
You could also try downloading those UCR datasets and doing adaptation between
some of them; though, I have not tried this.

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
~100 GiB of free space on your hard drive. Below I'll show you a way how I
set this up.

I'm on Arch, so I installed with
`sudo pacman -S postgresql`. Then since I don't have enough disk space on my root
partition, I told it to put the database in my home directory by running
`sudo -E systemctl edit postgresql.service` (replace "/pathto" with wherever
you want to put the database):

    [Service]
    Environment=PGROOT=/path/pgroot
    PIDFile=/pathto/pgroot/data/postmaster.pid
    ProtectHome=false

Created the directory, set to proper user, and then initialize database:

    pgroot="/pathto/pgroot"
    mkdir -p "$pgroot/data"
    sudo chown -R postgres:postgres "$pgroot"
    sudo -u postgres initdb -D "$pgroot/data"

Start the database:

    sudo systemctl daemon-reload
    sudo systemctl start postgresql

Create database and schema:

    sudo -u postgres createdb mimic
    psql -d mimic -c "CREATE SCHEMA mimiciii;" postgres

### Create the Database

Now we can create the database with the MIMIC-III scripts:

    datadir="../../../../mimic-iii/"
    cd datasets/mimic-code/buildmimic/postgres
    make mimic-gz datadir="$datadir"

### Process to Get Time-Series Data

You need to run all the processing to get time-series datasets out of the
database. First, make sure you have all the
[dependencies](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII) installed.
Next, tell it how to connect to your database by editing
`getConnection()` in *datasets/process-mimic-iii/Codes/mimic3_mvcv/utils.py*:

    def getConnection():
        return psycopg2.connect("dbname='mimic' user='postgres' password='postgres'")

Generate some more tables that are used during processing:

    dir="datasets/process-mimic-iii/Codes/mimic3_mvcv/sql_gen_17features_ts/"
    alias psqlf='psql -d mimic postgres -f'

    psqlf "$dir/gen_gcs_ts.sql"
    psqlf "$dir/gen_lab_ts.sql"
    psqlf "$dir/gen_pao2_fio2.sql"
    psqlf "$dir/gen_urine_output_ts.sql"
    psqlf "$dir/gen_vital_ts.sql"
    psqlf "$dir/gen_17features_first24h.sql"
    psqlf "$dir/gen_17features_first48h.sql"

Run a bunch of notebooks to generate the time-series data.

    dir="datasets/process-mimic-iii/Codes/mimic3_mvcv"
    alias jnb="jupyter nbconvert --execute --to notebook --inplace"

    jnb "$dir/0_createAdmissionList.ipynb"
    jnb "$dir/1_getItemIdList.ipynb"
    jnb "$dir/2_filterItemId_input.ipynb"
    jnb "$dir/3_filterItemId_output.ipynb"
    jnb "$dir/4_filterItemId_chart.ipynb"
    jnb "$dir/5_filterItemId_lab.ipynb"
    jnb "$dir/6_filterItemId_microbio.ipynb"
    jnb "$dir/7_filterItemId_prescript.ipynb"
    jnb "$dir/8_processing.ipynb"
    jnb "$dir/9_collect_mortality_labels.ipynb"
    jnb "$dir/9_getValidDataset.ipynb"
    jnb "$dir/10_get_17-features-processed(fromdb).ipynb"
    jnb "$dir/10_get_17-features-raw.ipynb"
    jnb "$dir/10_get_99plus-features-raw.ipynb"
    jnb "$dir/11_get_time_series_sample_17-features-processed_24hrs.ipynb"
    jnb "$dir/11_get_time_series_sample_17-features-processed_48hrs.ipynb"
    jnb "$dir/11_get_time_series_sample_17-features-raw_24hrs.ipynb"
    jnb "$dir/11_get_time_series_sample_17-features-raw_48hrs.ipynb"
    jnb "$dir/11_get_time_series_sample_99plus-features-raw_24hrs.ipynb"
    jnb "$dir/11_get_time_series_sample_99plus-features-raw_48hrs.ipynb"

### Create .tfrecord for TensorFlow

You can generate the .tfrecord files by running
`python3 datasets/generate_mimic_iii.py`.