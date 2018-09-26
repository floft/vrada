#!/bin/bash
#
# Download processed MIMIC-III data from high performance cluster
#
. kamiak_config.sh

# Note both have trailing slashes
from="$remotessh:$remotedir/"
to="$localdir/"

rsync -Pahuv "$from/$mimiccode/" "$to/$mimiccode/"
rsync -Pahuv "$from/$mimicdata/" "$to/$mimicdata/"
