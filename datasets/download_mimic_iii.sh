#!/bin/bash
#
# Download MIMIC-III (requires login info in .netrc, see ../README.md)
#
# They give a line for wget, but I don't want to hours days for it to download
# when I have a 1 GiB connection. This will download in a matter of minutes at
# 40 MiB/s.
#
dir="mimic-iii"
mkdir -p "$dir"

# Note: even this URL requires you be logged in
url="https://physionet.org/works/MIMICIIIClinicalDatabase/files/"

# First download the page with all the links
[[ ! -e $dir/list.html ]] && curl -n "$url" > "$dir/list.html"

# Then grab the csv.gz filenames and prepend them with the path to make it an
# absolute URL.
[[ ! -e $dir/list.txt ]] && grep "version_" < "$dir/list.html" | \
    sed -r 's#.*"([^"]+)".*#\1#g' | awk "\$0=\"$url\"\$0" > "$dir/list.txt"

# Then download all those files quickly with aria2c instead of wget
aria2c \
    --allow-overwrite=true \
    --continue=true \
    --file-allocation=none \
    --log-level=error \
    --max-tries=0 \
    --max-connection-per-server=3 \
    -j 5 \
    --max-file-not-found=5 \
    --min-split-size=5M \
    --no-conf \
    --remote-time=true \
    --summary-interval=60 \
    --timeout=5 \
    --check-integrity=true \
    -d "$dir" \
    -i "$dir/list.txt"

# Check md5sums
cd "$dir"
md5sum -c checksum_md5_zipped.txt
