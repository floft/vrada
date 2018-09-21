#
# Config file for running on high performance cluster with Slurm
#
dir="/data/vcea/matt.taylor/Projects/ras-object-detection/vrada"
program="VRADA.py"
modelFolder="vrada-sleep-models"
logFolder="vrada-sleep-logs"
imgFolder="vrada-sleep-images"
#compressedDataset="trivial.zip"
#dataset=("datasets/trivial")
compressedDataset="rfsleep.zip"
dataset=("datasets/RFSleep"/*.npy)

# For processing MIMIC-III dataset - directory for database
mimicdir="mimic-iii"
mimiccode="datasets/process-mimic-iii/Codes/mimic3_mvcv"
bindir="postgres/usr/local/pgsql/bin/" # where postgres and psql are
libdir="postgres/usr/local/pgsql/lib"

# Connecting to the remote server
# Note: this is rsync, so make sure all paths have a trailing slash
remotedir="$dir/"
remotessh="kamiak"
localdir="/home/garrett/Documents/School/18_Fall/Research/vrada/"
