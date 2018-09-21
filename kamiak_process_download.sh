#!/bin/bash
#
# Download processed MIMIC-III data from high performance cluster
#
. kamiak_config.sh

# Note both have trailing slashes
from="$remotessh:$remotedir/$mimiccode/"
to="$localdir/$mimiccode/"

rsync -Pahuv "$from" "$to"
