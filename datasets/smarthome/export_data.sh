#!/bin/bash
#
# Use the al.py script with the np.save addition to output the features and
# labels for each of the smart home datasets
#
# Usage: ./export_data.sh casas/*.al
#
if [[ -z $1 ]]; then
    echo "Usage: time ./export_data.sh ../watch/casas/*.al"
    exit 1
fi

echo "Updating sensor/activity lists"
sensor_names="$(cat "$@" | cut -d' ' -f3 | sort -u | tr '\n' ' ')"
activity_labels="$(cat "$@" | cut -d' ' -f5 | sort -u | tr '\n' ' ')"
sed -i "s#^sensors .*\$#sensors $sensor_names#g" "al.config"
sed -i "s#^activities .*\$#activities $activity_labels#g" "al.config"

. /scripts/threading
thread_init

for i; do
    echo "Processing $i"
    filename=$(basename -- "$i")
    filename="${filename%.*}"
    python2 al.py al.config "$i" "${filename}.hdf5" &
    thread_wait
done

thread_finish
