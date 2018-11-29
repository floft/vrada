#!/bin/bash
#
# Use the al.py script with the np.save addition to output the features and
# labels for each of the smart home datasets
#
# Usage: ./export_data.sh casas/*.al
#
echo "Updating sensor list"
sensor_names="$(cat "$@" | cut -d' ' -f3 | sort -u | tr '\n' ' ')"
sed -i "s#^sensors .*\$#sensors $sensor_names#g" "al.config"

for i; do
    ln -s "$i" "data"

    # Generate data.npy features and labels file
    echo "Processing $i"
    python2 al.py

    # Save data.npy
    if [[ -e data.npy ]]; then
        filename=$(basename -- "$i")
        filename="${filename%.*}"
        mv "data.npy" "${filename}.npy"
    else
        echo "Warning: no data.npy file found, an error occured"
    fi

    unlink "data"
done
