"""
Preprocess watch data

Since the watch data needs some cleanup and modification, we'll do that once,
save the output, and then load that into TensorFlow. This script does the
preprocessing.

This outputs files such as ihs95.npy. You can load this with
    d = np.load("ihs95.npy").item()
    d["name"] = "ihs95"
    d["times"].shape = (n,)
    d["features"].shape = (n,number of features)
    d["labels"] = (n,)
"""
import os
import re
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime

import pytz
from pytz import timezone

from pool import runPool

utc = timezone("UTC")
epoch = datetime.utcfromtimestamp(0).replace(tzinfo=utc)

def unix_time(dt):
    """ Calculate unix time, which is seconds since the "epoch", i.e. 1970 """
    # It will error if not in the same timezone, so convert to UTC for subtraction
    return (dt.astimezone(utc) - epoch).total_seconds()

def files_matching(dir_name, glob):
    """
    Find all files in dir_name matching glob and return the path and the ID for
    that data, which is provided in the filename before the first underscore or
    dot, e.g. "ihs95" if filename is "ihs95_day_2018...csv"

    If the name is unique, you could use files_matching_unique() instead,
    which indexes a dictionary by the name.

    Returns: [
        ('name1', PosixPath('path/to/name1_....ext')),
        ('name2', PosixPath('path/to/name2_....ext')),
        ...
    ]
    """
    files = list(pathlib.Path(dir_name).glob(glob))
    results = []
    regex = re.compile(r'^[a-zA-Z0-9]+')

    for f in files:
        name = list(regex.findall(str(f.stem)))
        assert len(name) == 1, "Could not determine name of file "+str(f)
        results.append((name[0], f))

    return results

def files_matching_unique(dir_name, glob):
    """
    Find all files in dir_name matching glob and return the path and the ID for
    that data, which is provided in the filename before the first underscore or
    dot, e.g. "ihs95" if filename is "ihs95_day_2018...csv"

    This version returns a dictionary indexed by the name, which assumes the
    name is unique. If it is not, use files_matching() instead.

    Returns:
        {
            'name1': PosixPath('path/to/name1_....ext'),
            'name2': PosixPath('path/to/name2_....ext'),
            ...
        }
    """
    files = list(pathlib.Path(dir_name).glob(glob))
    results = {}
    regex = re.compile(r'^[a-zA-Z0-9]+')

    for f in files:
        name = list(regex.findall(str(f.stem)))
        assert len(name) == 1, "Could not determine name of file "+str(f)
        assert name[0] not in results, "Name "+str(name[0])+" already found"
        results[name[0]] = f

    return results

def str_mapping(names, mapping):
    """
    Make replacements defined in mapping in all of the names
    """
    new_names = []

    for n in names:
        s = n

        for k, v in mapping.items():
            s = s.replace(k, v)

        new_names.append(s)

    return new_names

def replace_keys_inplace(data, mapping):
    """
    Given a mapping { "a1": "b1", "a2": "b2", ...}, replace each "aX" key name
    specified with the corresponding "bX" key name in the data dictionary.

    This modifies the data object itself.
    """
    for k, v in mapping.items():
        if k in data:
            data[v] = data[k]
            del data[k]

def parse_datetime(dt, timezone=utc):
    """
    Load the date/time from a couple possible formats into a Python datetime
    object and set the desired timezone

    If it doesn't fit a known format, this will throw a ValueError.
    """
    # Date time -- format documentation http://strftime.org/
    # Some apparently don't have the decimal and microseconds
    try:
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    finally:
        dt = dt.replace(tzinfo=timezone)

    return dt

def load_al_activity_times(file):
    """
    Load data in the activity learning format, where each line resembles:
        2012-01-01 13:55:55.999999 OutsideDoor OFF Other_Activity

    Returns:
        - [(datetime, sensor_name str, sensor_value str, activity_label str), ...]
        - sorted list of unique sensor names
        - sorted list of unique sensor values
        - sorted list of unique activity labels
        - the earliest time (datetime object)

    Note: probably could have written this with Pandas read_csv as well
    """
    data = []
    unique_names = []
    unique_values = []
    unique_labels = []
    earliest_time = None

    with open(file) as f:
        for i, line in enumerate(f):
            parts = line.strip().split(" ")
            assert len(parts) == 5, "Parts of a line "+len(parts)+" != 5"

            dt = parse_datetime(parts[0]+" "+parts[1])
            sensor_name = parts[2]
            sensor_value = parts[3]
            activity_label = parts[4]

            if earliest_time is None:
                earliest_time = dt
            elif dt < earliest_time:
                earliest_time = dt

            if sensor_name not in unique_names:
                unique_names.append(sensor_name)
            if sensor_value not in unique_values:
                unique_values.append(sensor_value)
            if activity_label not in unique_labels:
                unique_labels.append(activity_label)

            if dt is not None:
                data.append((dt, sensor_name, sensor_value, activity_label))

    unique_names = list(sorted(unique_names))
    unique_values = list(sorted(unique_values))
    unique_labels = list(sorted(unique_labels))

    return data, unique_names, unique_values, unique_labels, earliest_time

def al_to_numpy(file):
    """
    Convert the AL data into a numpy array of floats:
        [ [seconds since earliest time, label #], ... ]

    Returns:
        - numpy array
        - label array - e.g. label 0 in the numpy array corresponds to "car"
    """
    # Get data from file
    al_data, _, _, unique_labels, _ = load_al_activity_times(file)

    # Convert to a numpy array
    data = []

    for dt, _, _, label in al_data:
        data.append((unix_time(dt), unique_labels.index(label)))

    data = np.array(data, dtype=np.float64)

    # Sort by date since Brian said the files may not be in order
    data = data[data[:,0].argsort()]

    return data, unique_labels

def _load_home_process(name, home_file):
    """ Wrapper that includes name in the output """
    data, unique_labels = al_to_numpy(home_file)
    return (name, data, unique_labels)

def load_home_data(home_files):
    """
    Load the AL files given and extract each time and the activity occuring at
    that time

    Run on multiple cores since this is slow
    """
    commands = []
    for name, home_file in home_files.items():
        commands.append((name, home_file))

    results = runPool(_load_home_process, commands,
        desc="Load AL activity times")

    activity_labels = None
    home_data = {}

    for name, data, unique_labels in results:
        assert name not in home_data, "Name "+name+" is not unique"
        home_data[name] = data

        # Make sure we have consistent labels across files (if not, maybe take
        # the union and fix the label numbers?)
        if activity_labels is None:
            activity_labels = unique_labels
        else:
            assert activity_labels == unique_labels, "Labels differ between files"

    return home_data, activity_labels

def _parse_watch_time(time):
    """ Wrapper function including the time zone """
    # We know all the watch data was collected in this timezone
    return parse_datetime(time, timezone=timezone("US/Pacific"))

def load_watch_features(file):
    """
    Load the features from the watch CSV files (no labels though)
    """
    df = pd.read_csv(file)

    parsed_times = df["Sensor Data Time (Local)"].map(_parse_watch_time)
    times = parsed_times.values
    np_times = parsed_times.map(unix_time).values.astype(np.float64)

    features = df[[
        "Yaw (rad)",
        "Pitch (rad)",
        "Roll (rad)",
        "Rotation Rate X (rad/s)",
        "Rotation Rate Y (rad/s)",
        "Rotation Rate Z (rad/s)",
        "User Acceleration X (m/s^2)",
        "User Acceleration Y (m/s^2)",
        "User Acceleration Z (m/s^2)",
        "Latitude",
        "Longitude",
        "Altitude (m)",
        "Horizontal Accuracy (m)",
        "Vertical Accuracy (m)",
        "Course (deg)",
        "Speed (m/s)",
    ]].values.astype(np.float32)

    return times, np_times, features

def label_watch_data(times, np_times, features, corresponding_home_data,
    max_diff_secs=900):
    """
    Grab the nearest activity label from the smart home data that corresponds
    to this watch data

    times - datetime objects
    np_times - unix time, i.e. seconds since 1970

    If no label found within max_diff_secs seconds, it'll give a np.nan for the
    label instead -- later select only the labeled examples for training probably.
    """
    assert len(times) == len(features), "len(times) != len(features)"
    assert len(np_times) == len(features), "len(np_times) != len(features)"
    labels = np.zeros((len(times),), dtype=np.float32)

    for i in range(len(times)):
        # Values for this example
        t = np_times[i]

        # Find where we'd insert this timestamp into the home dataset, i.e.
        # right before and after this index will be the closest labels in the
        # smart home data
        #
        # Note: the array *must* be sorted for this to work, which we did in
        # al_to_numpy
        index = np.searchsorted(corresponding_home_data[:,0], t)

        # Figure out if it's the one before or at this index that is the closest
        # But, if 0 or N then there isn't a before or is only a before
        # respectively
        if index == 0:
            closest = index
            diff_secs = abs(t-corresponding_home_data[closest,0])
        elif index == len(corresponding_home_data):
            closest = index-1
            diff_secs = abs(t-corresponding_home_data[closest,0])
        else:
            t_before = corresponding_home_data[index-1,0]
            t_after = corresponding_home_data[index,0]

            if abs(t-t_before) < abs(t-t_after):
                closest = index-1
                diff_secs = abs(t-t_before)
            else:
                closest = index
                diff_secs = abs(t-t_after)

        # If not close enough, we'll skip labeling it and say NaN
        if diff_secs > max_diff_secs:
            labels[i] = np.nan
        # If it is close enough, get the label from the smart home data
        else:
            labels[i] = corresponding_home_data[closest, 1]

    return labels

def _load_watch_process(name, watch_file, corresponding_home_data):
    """ Wrapper that includes name in the output """
    times, np_times, features = load_watch_features(watch_file)
    labels = label_watch_data(times, np_times, features, corresponding_home_data)
    return (name, times, features, labels)

def load_watch_data(watch_files, home_data):
    """
    Load the watch data from CSV files

    Run on multiple cores since this is slow
    """
    commands = []
    for name, watch_file in watch_files:
        commands.append((name, watch_file, home_data[name]))

    results = runPool(_load_watch_process, commands,
        desc="Load watch times and features")

    # Join the results that have the same name (i.e. each will have a day and a
    # night one)
    datasets = {}

    for name, times, features, labels in results:
        if name not in datasets:
            datasets[name] = (times, features, labels)
        else:
            old_times = datasets[name][0]
            old_features = datasets[name][1]
            old_labels = datasets[name][2]

            datasets[name] = (
                np.hstack([old_times, times]),
                np.vstack([old_features, features]),
                np.hstack([old_labels, labels]),
            )

    return datasets

if __name__ == "__main__":
    dataset_folder = "datasets/watch/"

    # List of all data files from both watch and smart home
    watch_files = files_matching(os.path.join(dataset_folder, "watch"), "*.csv")
    home_files = files_matching_unique(os.path.join(dataset_folder, "casas"), "*.al")

    # They use different names, so for each watch file we'll want to be able
    # to get the corresponding smart home data. Here we will change the names.
    replace_keys_inplace(home_files, {
        "rw101": "ihs107",
        "rw106": "ihs108",
        "rw105": "ihs114",
    })

    # Load and preprocess home data -- indexed by name
    home_data, activity_labels = load_home_data(home_files)

    # Load the watch data and get the corresponding activity labels from
    # the nearest activity label in the home's labeled data
    watch_datasets = load_watch_data(watch_files, home_data)

    # Save the results
    for name, (times, features, labels) in watch_datasets.items():
        np.save(os.path.join(dataset_folder, name+".npy"), {
            "name": name,
            "times": times,
            "features": features,
            "labels": labels,
        })
