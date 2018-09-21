#!/bin/bash
#
# Upload files to high performance cluster
#
. kamiak_config.sh

# Note both have trailing slashes
from="$localdir"
to="$remotessh:$remotedir"

# Compress dataset
[[ ! -f $compressedDataset ]] && zip -r "$compressedDataset" "${dataset[@]}"

# Make SLURM log folder
ssh "$remotessh" "mkdir -p \"$remotedir/slurm_logs\""

# Copy only select files
rsync -Pahuv --include="./" --include="*.py" --include="*.sh" --include="*.srun" \
    --include="$compressedDataset" --exclude="*" "$from" "$to"
