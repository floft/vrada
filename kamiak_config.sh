#
# Config file for running on high performance cluster with Slurm
#
dir="/data/vcea/garrett.wilson/vrada"
program="VRADA.py"
modelFolder="vrada-mimic-icd9-models"
logFolder="vrada-mimic-icd9-logs"
#compressedDataset="trivial.zip"
#dataset=("datasets/trivial")
#compressedDataset="rfsleep.zip"
#dataset=("datasets/RFSleep"/*.npy)
#compressedDataset="mimiciii.zip"
#dataset=("datasets/process-mimic-iii/Data/admdata_17f/24hrs/series/"/{imputed-normed-ep_1_24.npz,5-folds.npz})
compressedDataset="mimiciii-icd9.zip"
dataset=("datasets/process-mimic-iii/Data/admdata_99p/48hrs_raw/series/"/{imputed-normed-ep_1_48.npz,5-folds.npz})

# For processing MIMIC-III dataset - directory for database
mimicdir="mimic-iii"
mimiccode="datasets/process-mimic-iii/Codes/mimic3_mvcv"
mimicdata="datasets/process-mimic-iii/Data"
bindir="postgres/usr/local/pgsql/bin/" # where postgres and psql are
libdir="postgres/usr/local/pgsql/lib"

# Connecting to the remote server
# Note: this is rsync, so make sure all paths have a trailing slash
remotedir="$dir/"
remotessh="kamiak"
localdir="/home/garrett/Documents/School/18_Fall/Research/vrada/"
