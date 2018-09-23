"""
Load data

Functions to load the data into TensorFlow
"""
import math
import random
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf

# Not-so-pretty code to feed data to TensorFlow.
class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created.
    https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0"""
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iter_init_func = None

    def after_create_session(self, sess, coord):
        """Initialize the iterator after the session has been created."""
        self.iter_init_func(sess)

def _get_input_fn(features, labels, batch_size, evaluation=False, buffer_size=5000):
    iter_init_hook = IteratorInitializerHook()

    def input_fn():
        # Input images using placeholders to reduce memory usage
        features_placeholder = tf.placeholder(features.dtype, features.shape)
        labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

        if evaluation:
            dataset = dataset.batch(batch_size)
        else:
            dataset = dataset.repeat().shuffle(buffer_size).batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        next_data_batch, next_label_batch = iterator.get_next()

        # Need to initialize iterator after creating a session in the estimator
        iter_init_hook.iter_init_func = lambda sess: sess.run(iterator.initializer,
                feed_dict={features_placeholder: features, labels_placeholder: labels})

        return next_data_batch, next_label_batch
    return input_fn, iter_init_hook

# Load a time-series dataset. This is set up to load data in the format of the
# UCR time-series datasets (http://www.cs.ucr.edu/~eamonn/time_series_data/).
# Or, see the generate_trivial_datasets.py for a trivial dataset.
#
# Also runs through one_hot
def load_data(filename):
    """
    Load CSV files in UCR time-series data format

    Returns:
        data - numpy array with data of shape (num_examples, num_features)
        labels - numpy array with labels of shape: (num_examples, 1)
    """
    df = pd.read_csv(filename, header=None)
    df_data = df.drop(0, axis=1).values.astype(np.float32)
    df_labels = df.loc[:, df.columns == 0].values.astype(np.uint8)
    return df_data, df_labels

def one_hot(x, y, num_classes, index_one=False):
    """
    We want x to be floating point and of dimension [time_steps,num_features]
    where num_features is at least 1. If only a 1D array, then expand dimensions
    to make it [time_steps, 1].

    Also, we want y to be one-hot encoded. Though, note that for the UCR datasets
    (and my synthetic ones that I used the UCR dataset format for), it's indexed
    by 1 not 0, so we subtract one from the index. But, for most other datasets,
    it's 0-indexed.
    """
    # Floating point
    x = x.astype(np.float32)

    # For if we only have one feature,
    # [batch_size, time_steps] --> [batch_size, time_steps, 1]
    if len(x.shape) < 3:
        x = np.expand_dims(x, axis=2)

    # One-hot encoded
    if index_one:
        y = np.eye(num_classes, dtype=np.float32)[np.squeeze(y).astype(np.int32) - 1]
    else:
        y = np.eye(num_classes, dtype=np.float32)[np.squeeze(y).astype(np.int32)]

    return x, y

def tf_domain_labels(label, batch_size):
    """ Generate one-hot encoded labels for which domain data is from (using TensorFlow) """
    return tf.tile(tf.one_hot([0], depth=2), [batch_size,1])

def domain_labels(label, batch_size):
    """ Generate one-hot encoded labels for which domain data is from (using numpy) """
    return np.tile(np.eye(2)[label], [batch_size,1])

def shuffle_together(a, b, seed=None):
    """ Shuffle two lists in unison https://stackoverflow.com/a/13343383/2698494 """
    assert len(a) == len(b), "a and b must be the same length"
    rand = random.Random(seed)
    combined = list(zip(a, b))
    rand.shuffle(combined)
    return zip(*combined)

def shuffle_together_np(a, b, seed=None):
    """ Shuffle two numpy arrays together https://stackoverflow.com/a/4602224/2698494"""
    assert len(a) == len(b), "a and b must be the same length"
    rand = np.random.RandomState(seed)
    p = rand.permutation(len(a))
    return a[p], b[p]

# Load sleep paper datasets (RF data)
def load_data_sleep(dir_name, domain_a_percent=0.7, train_percent=0.7, seed=0):
    """
    Loads sleep RF data files in dir_name/*.npy
    Then split into training/testing sets using the specified seed for repeatability.

    We'll split the data twice. First, we split into domain A and domain B based
    on subjects (so no subject will be in both domains). Then, we concatenate all
    the data for each domain and randomly split into training and testing sets.

    Notes:
        - RF data is 30 seconds of data sampled at 25 samples per second, thus
          750 samples. For each of these sets of 750 samples there is a stage
          label.
        - The RF data is complex, so we'll split the complex 5 features into
          the 5 real and then 5 imaginary components to end up with 10 features.
    """
    #
    # Get data from data files grouped by subject
    #
    files = pathlib.Path(dir_name).glob("*.npy")
    subject_x = {}
    subject_y = {}

    for f in files:
        # Extract data from file
        exp_data = np.load(f).item()
        subject = exp_data['subject']
        stage_labels = exp_data['stage']
        rf = exp_data['rf']

        # Split 5 complex features into 5 real and 5 imaginary, i.e. now we
        # have 10 features
        rf = np.vstack([np.real(rf), np.imag(rf)])

        assert stage_labels.shape[0]*750 == rf.shape[-1], \
            "If stage labels is of shape (n) then rf should be of shape (5, 750n)"

        # Reshape and transpose into desired format
        x = np.transpose(np.reshape(rf, (rf.shape[0], -1, stage_labels.shape[0])))

        # Drop those that have a label other than 0-5 (sleep stages) since
        # label 6 means "no signal" and 9 means "error"
        no_error = stage_labels < 6
        x = x[no_error]
        stage_labels = stage_labels[no_error]

        assert x.shape[0] == stage_labels.shape[0], \
            "Incorrect first dimension of x (not length of stage labels)"
        assert x.shape[1] == 750, \
            "Incorrect second dimension of x (not 750)"
        assert x.shape[2] == 10, \
            "Incorrect third dimension of x (not 10)"

        # Group data by subject, stacking new data at bottom of old data
        if subject not in subject_x:
            subject_x[subject] = x
            subject_y[subject] = stage_labels
        else:
            subject_x[subject] = np.vstack([subject_x[subject], x])
            subject_y[subject] = np.hstack([subject_y[subject], stage_labels])

    #
    # Split subjects into training vs. testing and concatenate all the
    # data into training and testing sets
    #
    # Shuffle the subject ordering (using our seed for repeatability)
    xs = list(subject_x.values())
    ys = list(subject_y.values())
    xs, ys = shuffle_together(xs, ys, seed)

    # Split into two domains such that no subject is in both
    domain_end = math.ceil(domain_a_percent*len(xs))

    domain_a_x = xs[:domain_end]
    domain_b_x = xs[domain_end:]

    domain_a_y = ys[:domain_end]
    domain_b_y = ys[domain_end:]

    # Concatenate all the data from subjects
    a_x = np.vstack(domain_a_x)
    a_y = np.hstack(domain_a_y).astype(np.int32)
    b_x = np.vstack(domain_b_x)
    b_y = np.hstack(domain_b_y).astype(np.int32)

    # Shuffle data, using our seed
    a_x, a_y = shuffle_together_np(a_x, a_y, seed+1)
    b_x, b_y = shuffle_together_np(b_x, b_y, seed+2)

    # Split into training and testing sets
    training_end_a = math.ceil(train_percent*len(a_y))
    training_end_b = math.ceil(train_percent*len(b_y))

    train_data_a = a_x[:training_end_a]
    train_data_b = b_x[:training_end_b]
    test_data_a = a_x[training_end_a:]
    test_data_b = b_x[training_end_b:]

    train_labels_a = a_y[:training_end_a]
    train_labels_b = b_y[:training_end_b]
    test_labels_a = a_y[training_end_a:]
    test_labels_b = b_y[training_end_b:]

    return train_data_a, train_labels_a, \
        test_data_a, test_labels_a, \
        train_data_b, train_labels_b, \
        test_data_b, test_labels_b
