#!/bin/bash
#
# When downloading the remote-created models with ./kamiak_download.sh
# all the paths point to the remote location, so they won't load the
# model locally. This replaces the remote path with the local path.
#
. kamiak_config.sh

sed -i "s#$remotedir#$localdir#g" "$modelFolder/checkpoint"