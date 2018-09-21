#!/bin/bash
#
# Since processing takes a long time even on the high performance cluster,
# monitor the progress to get an idea how long it'll take.
#
. kamiak_config.sh

cd "$mimiccode"

# Percentage done
num="$(ls admdata/log | wc -l)"
total=58976

# How long it's currently been running
running="$(SQUEUE_FORMAT="%M" squeue -A taylor -n mimic | tail -n +2)"
hours="$(cut -d':' -f1 <<< "$running")"
minutes="$(cut -d':' -f2 <<< "$running")"

# How long we have left -- it takes around 18 minutes to get to where it's
# starting to output the .npy files, so ignore those 18 minutes
days_left="$(bc <<< "scale=5;($total-$num)/($num/($hours*60+$minutes-18))/60/24")"

echo "Completed: $num"
echo "Days left: $days_left"
