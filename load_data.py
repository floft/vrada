"""
Load data

Functions to load the data into TensorFlow
"""
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

def one_hot(x, y, num_classes):
    """ Correct dimensions, type, and one-hot encode """
    x = np.expand_dims(x, axis=2).astype(np.float32)
    y = np.eye(num_classes)[np.squeeze(y).astype(np.uint8) - 1] # one-hot encode
    return x, y

def tf_domain_labels(label, batch_size):
    """ Generate one-hot encoded labels for which domain data is from (using TensorFlow) """
    return tf.tile(tf.one_hot([0], depth=2), [batch_size,1])

def domain_labels(label, batch_size):
    """ Generate one-hot encoded labels for which domain data is from (using numpy) """
    return np.tile(np.eye(2)[label], [batch_size,1])