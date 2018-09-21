"""
VRADA implementation

See the paper: https://openreview.net/pdf?id=rk9eAFcxg
See coauthor blog post: https://wcarvalho.github.io/research/2017/04/23/vrada/
"""
import os
import re
import time
import argparse
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#from tensorflow.contrib.tensorboard.plugins import projector

# Due to this being run on Kamiak, that doesn't have _tkinter, we have to set a
# different backend otherwise it'll error
# https://stackoverflow.com/a/40931739/2698494
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

from plot import plot_embedding, plot_random_time_series
from model import build_lstm, build_vrnn
from load_data import IteratorInitializerHook, \
    load_data, one_hot, \
    domain_labels, _get_input_fn, \
    load_data_sleep

def evaluation_accuracy(sess,
    eval_input_hook_a, eval_input_hook_b,
    next_data_batch_test_a, next_labels_batch_test_a,
    next_data_batch_test_b, next_labels_batch_test_b,
    task_accuracy_sum, domain_accuracy_sum,
    source_domain, target_domain,
    x, y, domain, keep_prob, training,
    batch_size):
    """
    Run all the evaluation data to calculate accuracy

    We may not be able to fit all the evaluation data into memory, so we'll
    loop over it in batches and at the end calculate the accuracy.
    """
    # Evaluation set in batches (since it may not fit into memory
    # all at once)
    task_a = 0
    task_b = 0
    domain_a = 0
    domain_b = 0
    a_total = 0
    b_total = 0

    # Reinitialize the evaluation batch initializers so we start at
    # the beginning of the evaluation data again
    eval_input_hook_a.iter_init_func(sess)
    eval_input_hook_b.iter_init_func(sess)

    while True:
        try:
            # Get next evaluation batch
            eval_data_a, eval_labels_a, eval_data_b, eval_labels_b = sess.run([
                next_data_batch_test_a, next_labels_batch_test_a,
                next_data_batch_test_b, next_labels_batch_test_b,
            ])

            # For simplicity, don't use the last part of the batch if
            # we won't have a full batch
            if eval_data_a.shape[0] != batch_size or eval_data_b.shape[0] != batch_size:
                break

            # Log summaries run on the evaluation/validation data
            batch_task_a, batch_domain_a = sess.run(
                [task_accuracy_sum, domain_accuracy_sum], feed_dict={
                x: eval_data_a, y: eval_labels_a, domain: source_domain,
                keep_prob: 1.0, training: False
            })
            batch_task_b, batch_domain_b = sess.run(
                [task_accuracy_sum, domain_accuracy_sum], feed_dict={
                x: eval_data_b, y: eval_labels_b, domain: target_domain,
                keep_prob: 1.0, training: False
            })

            task_a += batch_task_a
            task_b += batch_task_b
            domain_a += batch_domain_a
            domain_b += batch_domain_b
            a_total += eval_data_a.shape[0]
            b_total += eval_data_b.shape[0]
        except tf.errors.OutOfRangeError:
            break

    task_a_accuracy = task_a / a_total
    domain_a_accuracy = domain_a / a_total
    task_b_accuracy = task_b / b_total
    domain_b_accuracy = domain_b / b_total

    return task_a_accuracy, domain_a_accuracy, \
        task_b_accuracy, domain_b_accuracy

def train(data_info,
        features_a, labels_a, test_features_a, test_labels_a,
        features_b, labels_b, test_features_b, test_labels_b,
        model_func=build_lstm,
        batch_size=128,
        num_steps=100000,
        learning_rate=0.0003,
        lr_multiplier=1,
        dropout_keep_prob=0.8,
        model_dir="models",
        log_dir="logs",
        img_dir="images",
        embedding_prefix=None,
        model_save_steps=1000,
        log_save_steps=50,
        log_extra_save_steps=250,
        adaptation=True):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Data stats
    time_steps, num_features, num_classes = data_info

    # For adaptation, we'll be concatenating together half source and half target
    # data, so to keep the batch_size about the same, we'll cut it in half
    if adaptation:
        batch_size = batch_size // 2

    # Input training data
    with tf.variable_scope("training_data_a"):
        input_fn_a, input_hook_a = _get_input_fn(features_a, labels_a, batch_size)
        next_data_batch_a, next_labels_batch_a = input_fn_a()
    with tf.variable_scope("training_data_b"):
        input_fn_b, input_hook_b = _get_input_fn(features_b, labels_b, batch_size)
        next_data_batch_b, next_labels_batch_b = input_fn_b()

    # Load all the test data in one batch (we'll assume test set is small for now)
    with tf.variable_scope("evaluation_data_a"):
        eval_input_fn_a, eval_input_hook_a = _get_input_fn(
                test_features_a, test_labels_a, batch_size, evaluation=True)
        next_data_batch_test_a, next_labels_batch_test_a = eval_input_fn_a()
    with tf.variable_scope("evaluation_data_b"):
        eval_input_fn_b, eval_input_hook_b = _get_input_fn(
                test_features_b, test_labels_b, batch_size, evaluation=True)
        next_data_batch_test_b, next_labels_batch_test_b = eval_input_fn_b()

    # Inputs
    keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob') # for dropout
    x = tf.placeholder(tf.float32, [None, time_steps, num_features], name='x') # input data
    domain = tf.placeholder(tf.float32, [None, 2], name='domain') # which domain
    y = tf.placeholder(tf.float32, [None, num_classes], name='y') # class 1, 2, etc. one-hot
    training = tf.placeholder(tf.bool, name='training') # whether we're training (batch norm)
    grl_lambda = tf.placeholder_with_default(1.0, shape=(), name='grl_lambda') # gradient multiplier for GRL
    lr = tf.placeholder(tf.float32, (), name='learning_rate')

    # Source domain will be [[1,0], [1,0], ...] and target domain [[0,1], [0,1], ...]
    source_domain = domain_labels(0, batch_size)
    target_domain = domain_labels(1, batch_size)

    # Model, loss, feature extractor output -- e.g. using build_lstm or build_vrnn
    #
    # Optionally also returns additional summaries to log, e.g. loss components
    task_classifier, domain_classifier, total_loss, \
    feature_extractor, model_summaries, extra_model_outputs = \
        model_func(x, y, domain, grl_lambda, keep_prob, training,
            num_classes, num_features, adaptation)

    # Accuracy of the classifiers -- https://stackoverflow.com/a/42608050/2698494
    #
    # Note: we also calculate the *sum* (not just the mean), since we need to
    # run these multiple times if we can't fit the entire validation set into
    # memory. Then afterwards we can divide by the size of the validation set.
    with tf.variable_scope("task_accuracy"):
        equals = tf.cast(
            tf.equal(tf.argmax(y, axis=-1), tf.argmax(task_classifier, axis=-1)),
        tf.float32)
        task_accuracy_sum = tf.reduce_sum(equals)
        task_accuracy = tf.reduce_mean(equals)
    with tf.variable_scope("domain_accuracy"):
        equals = tf.cast(
            tf.equal(tf.argmax(domain, axis=-1), tf.argmax(domain_classifier, axis=-1)),
        tf.float32)
        domain_accuracy_sum = tf.reduce_sum(equals)
        domain_accuracy = tf.reduce_mean(equals)

    # Get variables of model - needed if we train in two steps
    variables = tf.trainable_variables()
    rnn_vars = [v for v in variables if 'rnn_model' in v.name]
    feature_extractor_vars = [v for v in variables if 'feature_extractor' in v.name]
    task_classifier_vars = [v for v in variables if 'task_classifier' in v.name]
    domain_classifier_vars = [v for v in variables if 'domain_classifier' in v.name]

    # Optimizer - update ops for batch norm layers
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(lr)
        train_all = optimizer.minimize(total_loss)
        train_notdomain = optimizer.minimize(total_loss,
            var_list=rnn_vars+feature_extractor_vars+task_classifier_vars)

        if adaptation:
            train_domain = optimizer.minimize(total_loss,
                var_list=domain_classifier_vars)

    # For making sure batch norm is working -- moving averages
    # global_variables = tf.global_variables()
    # moving_batch_vars = [v for v in global_variables if 'moving_' in v.name]

    # for v in moving_batch_vars:
    #     model_summaries.append(
    #         tf.summary.histogram(v.name.replace(":0",""), v)
    #     )

    # Summaries - training and evaluation for both domains A and B
    training_summaries_a = tf.summary.merge([
        tf.summary.scalar("loss/total_loss", total_loss),
        tf.summary.scalar("accuracy/task/source/training", task_accuracy),
        tf.summary.scalar("accuracy/domain/source/training", domain_accuracy),
    ])
    training_summaries_extra_a = tf.summary.merge(model_summaries)
    training_summaries_b = tf.summary.merge([
        tf.summary.scalar("accuracy/task/target/training", task_accuracy),
        tf.summary.scalar("accuracy/domain/target/training", domain_accuracy)
    ])

    # Allow restoring global_step from past run
    global_step = tf.Variable(0, name="global_step", trainable=False)
    inc_global_step = tf.assign_add(global_step, 1, name='incr_global_step')

    # Keep track of state and summaries
    saver = tf.train.Saver(max_to_keep=num_steps)
    saver_hook = tf.train.CheckpointSaverHook(model_dir,
            save_steps=model_save_steps, saver=saver)
    writer = tf.summary.FileWriter(log_dir)

    # Start training
    with tf.train.SingularMonitoredSession(checkpoint_dir=model_dir, hooks=[
                input_hook_a, input_hook_b,
                saver_hook
            ]) as sess:

        for i in range(sess.run(global_step),num_steps+1):
            if i == 0:
                writer.add_graph(sess.graph)

            # GRL schedule and learning rate schedule from DANN paper
            grl_lambda_value = 2/(1+np.exp(-10*(i/(num_steps+1))))-1
            #lr_value = 0.001/(1.0+10*i/(num_steps+1))**0.75
            lr_value = learning_rate

            t = time.time()
            step = sess.run(inc_global_step)

            # Get data for this iteration
            data_batch_a, labels_batch_a, data_batch_b, labels_batch_b = sess.run([
                next_data_batch_a, next_labels_batch_a,
                next_data_batch_b, next_labels_batch_b,
            ])

            if adaptation:
                # Concatenate for adaptation - concatenate source labels with all-zero
                # labels for target since we can't use the target labels during
                # unsupervised domain adaptation
                combined_x = np.concatenate((data_batch_a, data_batch_b), axis=0)
                combined_labels = np.concatenate((labels_batch_a, np.zeros(labels_batch_b.shape)), axis=0)
                combined_domain = np.concatenate((source_domain, target_domain), axis=0)

                # Train everything in one step which should give a similar result
                sess.run(train_all, feed_dict={
                    x: combined_x, y: combined_labels, domain: combined_domain,
                    grl_lambda: grl_lambda_value,
                    keep_prob: dropout_keep_prob, lr: lr_value, training: True
                })

                # Update domain more
                #
                # Depending on the num_steps, your learning rate, etc. it may be
                # beneficial to increase the learning rate here to e.g. 10*lr
                # or 100*lr. This may also depend on your dataset though.
                sess.run(train_domain, feed_dict={
                    x: combined_x, y: combined_labels, domain: combined_domain,
                    grl_lambda: 0.0,
                    keep_prob: dropout_keep_prob, lr: lr_multiplier*lr_value, training: True
                })
            else:
                # Train task classifier on source domain to be correct
                sess.run(train_notdomain, feed_dict={
                    x: data_batch_a, y: labels_batch_a,
                    keep_prob: dropout_keep_prob, lr: lr_value, training: True
                })

            t = time.time() - t

            if i%log_save_steps == 0:
                # Log the step time
                summ = tf.Summary(value=[
                    tf.Summary.Value(tag="step_time", simple_value=t)
                ])
                writer.add_summary(summ, step)

                # Log summaries run on the training data
                summ = sess.run(training_summaries_a, feed_dict={
                    x: data_batch_a, y: labels_batch_a, domain: source_domain,
                    keep_prob: 1.0, training: False
                })
                writer.add_summary(summ, step)
                summ = sess.run(training_summaries_b, feed_dict={
                    x: data_batch_b, y: labels_batch_b, domain: target_domain,
                    keep_prob: 1.0, training: False
                })
                writer.add_summary(summ, step)

            # Larger stuff like weights and evaluation occasionally
            if i%log_extra_save_steps == 0:
                summ = sess.run(training_summaries_extra_a, feed_dict={
                    x: data_batch_a, y: labels_batch_a, domain: source_domain,
                    keep_prob: 1.0, training: False
                })
                writer.add_summary(summ, step)

                task_a_accuracy, domain_a_accuracy, \
                task_b_accuracy, domain_b_accuracy = evaluation_accuracy(sess,
                    eval_input_hook_a, eval_input_hook_b,
                    next_data_batch_test_a, next_labels_batch_test_a,
                    next_data_batch_test_b, next_labels_batch_test_b,
                    task_accuracy_sum, domain_accuracy_sum,
                    source_domain, target_domain,
                    x, y, domain, keep_prob, training,
                    batch_size)

                task_source_val = tf.Summary(value=[tf.Summary.Value(
                    tag="accuracy/task/source/validation",
                    simple_value=task_a_accuracy
                )])
                domain_source_val = tf.Summary(value=[tf.Summary.Value(
                    tag="accuracy/domain/source/validation",
                    simple_value=domain_a_accuracy
                )])
                task_target_val = tf.Summary(value=[tf.Summary.Value(
                    tag="accuracy/task/target/validation",
                    simple_value=task_b_accuracy
                )])
                domain_target_val = tf.Summary(value=[tf.Summary.Value(
                    tag="accuracy/domain/target/validation",
                    simple_value=domain_b_accuracy
                )])
                writer.add_summary(task_source_val, step)
                writer.add_summary(domain_source_val, step)
                writer.add_summary(task_target_val, step)
                writer.add_summary(domain_target_val, step)

                # Make sure we write to disk before too long so we can monitor live in
                # TensorBoard. If it's too delayed we won't be able to detect problems
                # for a long time.
                writer.flush()

        writer.flush()

        # Output t-SNE after we've trained everything on the evaluation data
        #
        # Maybe in the future it would be cool to use TensorFlow's projector in TensorBoard
        # https://medium.com/@vegi/visualizing-higher-dimensional-data-using-t-sne-on-tensorboard-7dbf22682cf2
        if embedding_prefix is not None:
            """
            if os.path.exists(tsne_filename+'_tsne_fit.npy'):
                print("Note: generating t-SNE plot using using pre-existing embedding")
                tsne = np.load(tsne_filename+'_tsne_fit.npy')
                pca = np.load(tsne_filename+'_pca_fit.npy')
            else:
                print("Note: did not find t-SNE weights -- recreating")

            np.save(tsne_filename+'_tsne_fit', tsne)
            np.save(tsne_filename+'_pca_fit', pca)
            """
            # Get the first batch of evaluation data to use for these plots
            eval_input_hook_a.iter_init_func(sess)
            eval_input_hook_b.iter_init_func(sess)
            eval_data_a, eval_labels_a, eval_data_b, eval_labels_b = sess.run([
                next_data_batch_test_a, next_labels_batch_test_a,
                next_data_batch_test_b, next_labels_batch_test_b,
            ])

            combined_x = np.concatenate((eval_data_a, eval_data_b), axis=0)
            combined_labels = np.concatenate((eval_labels_a, eval_labels_b), axis=0)
            combined_domain = np.concatenate((source_domain, target_domain), axis=0)

            embedding = sess.run(feature_extractor, feed_dict={
                x: combined_x, keep_prob: 1.0, training: False
            })

            tsne = TSNE(n_components=2, init='pca', n_iter=3000).fit_transform(embedding)
            pca = PCA(n_components=2).fit_transform(embedding)

            if adaptation:
                title = "Domain Adaptation"
            else:
                title = "No Adaptation"

            plot_embedding(tsne, combined_labels.argmax(1), combined_domain.argmax(1),
                title=title + " - t-SNE", filename=os.path.join(img_dir, embedding_prefix+"_tsne.png"))
            plot_embedding(pca, combined_labels.argmax(1), combined_domain.argmax(1),
                title=title + " - PCA", filename=os.path.join(img_dir, embedding_prefix+"_pca.png"))

            # Output time-series "reconstructions" from our generator (if VRNN)
            if extra_model_outputs is not None:
                # We'll get the decoder's mu and sigma from the evaluation/validation set since
                # it's much larger than the training batches
                mu, sigma = sess.run(extra_model_outputs, feed_dict={
                    x: eval_data_a, keep_prob: 1.0, training: False
                })

                plot_random_time_series(mu, sigma, title='VRNN Samples (source domain)',
                    filename=os.path.join(img_dir, embedding_prefix+"_reconstruction_a.png"))

                mu, sigma = sess.run(extra_model_outputs, feed_dict={
                    x: eval_data_b, keep_prob: 1.0, training: False
                })

                plot_random_time_series(mu, sigma, title='VRNN Samples (target domain)',
                    filename=os.path.join(img_dir, embedding_prefix+"_reconstruction_b.png"))

def last_modified_number(dir_name, glob):
    """
    Looks in dir_name at all files matching glob and takes number
    from the one last modified
    """
    files = pathlib.Path(dir_name).glob(glob)
    files = sorted(files, key=lambda cp:cp.stat().st_mtime)

    if len(files) > 0:
        # Get number from filename
        regex = re.compile(r'\d+')
        numbers = [int(x) for x in regex.findall(str(files[-1]))]
        assert len(numbers) == 1, "Could not determine number from last modified file"
        last = numbers[0]

        return last

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', default="models", type=str,
        help="Directory for saving model files")
    parser.add_argument('--logdir', default="logs", type=str,
        help="Directory for saving log files")
    parser.add_argument('--imgdir', default="images", type=str,
        help="Directory for saving image files")
    parser.add_argument('--lstm', dest='lstm', action='store_true',
        help="Run LSTM model")
    parser.add_argument('--vrnn', dest='vrnn', action='store_true',
        help="Run VRNN model")
    parser.add_argument('--lstm-da', dest='lstm_da', action='store_true',
        help="Run LSTM-DA model")
    parser.add_argument('--vrnn-da', dest='vrnn_da', action='store_true',
        help="Run VRNN-DA model")
    parser.add_argument('--mimic', dest='mimic', action='store_true',
        help="Run on the MIMIC-III dataset")
    parser.add_argument('--sleep', dest='sleep', action='store_true',
        help="Run on the RF sleep stage dataset")
    parser.add_argument('--trivial', dest='trivial', action='store_true',
        help="Run on the trivial synthetic dataset")
    parser.add_argument('--debug', dest='debug', action='store_true',
        help="Start new log/model/images rather than continuing from previous run")
    parser.add_argument('--no-debug', dest='debug', action='store_false',
        help="Do not increment model/log/image count each run (default)")
    parser.add_argument('--lrmult', default=1, type=int,
        help="Integer multiplier for extra discriminator training learning rate (default 1)")
    parser.set_defaults(
        lstm=False, vrnn=False, lstm_da=False, vrnn_da=False,
        mimic=False, sleep=False, trivial=False,
        debug=False)
    args = parser.parse_args()

    # Load datasets - domains A & B
    assert args.mimic + args.sleep + args.trivial == 1, \
        "Must specify exactly one dataset to use"

    if args.trivial:
        # Change in y-intercept
        train_data_a, train_labels_a = load_data("datasets/trivial/positive_slope_TRAIN")
        test_data_a, test_labels_a = load_data("datasets/trivial/positive_slope_TEST")
        train_data_b, train_labels_b = load_data("datasets/trivial/positive_slope_low_TRAIN")
        test_data_b, test_labels_b = load_data("datasets/trivial/positive_slope_low_TEST")

        # Information about dataset - same for both domains on these datasets
        index_one = True # Labels start from 1 not 0
        num_features = 1
        time_steps = train_data_a.shape[1]
        num_classes = len(np.unique(train_labels_a))
        data_info = (time_steps, num_features, num_classes)
    elif args.sleep:
        train_data_a, train_labels_a, \
        test_data_a, test_labels_a, \
        train_data_b, train_labels_b, \
        test_data_b, test_labels_b = load_data_sleep("datasets/RFSleep")

        # Information about dataset
        index_one = False # Labels start from 0
        num_features = train_data_a.shape[2]
        time_steps = train_data_a.shape[1]
        num_classes = len(np.unique(train_labels_a))
        data_info = (time_steps, num_features, num_classes)
    elif args.mimic:
        index_one = False
        raise NotImplementedError()

    # One-hot encoding
    train_data_a, train_labels_a = one_hot(train_data_a, train_labels_a, num_classes, index_one)
    test_data_a, test_labels_a = one_hot(test_data_a, test_labels_a, num_classes, index_one)
    train_data_b, train_labels_b = one_hot(train_data_b, train_labels_b, num_classes, index_one)
    test_data_b, test_labels_b = one_hot(test_data_b, test_labels_b, num_classes, index_one)

    # Train model using selected dataset and method
    assert args.lstm + args.vrnn + args.lstm_da + args.vrnn_da == 1, \
        "Must specify exactly one method to run"

    if args.lstm:
        prefix = "lstm"
        adaptation = False
        model_func = build_lstm
    elif args.vrnn:
        prefix = "vrnn"
        adaptation = False
        model_func = build_vrnn
    elif args.lstm_da:
        prefix = "lstm-da"
        adaptation = True
        model_func = build_lstm
    elif args.vrnn_da:
        prefix = "vrnn-da"
        adaptation = True
        model_func = build_vrnn

    if args.debug:
        attempt = last_modified_number(args.logdir, prefix+"*")
        attempt = attempt+1 if attempt is not None else 1
        print("Attempt:", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(args.modeldir, prefix)
        log_dir = os.path.join(args.logdir, prefix)
    else:
        model_dir = args.modeldir
        log_dir = args.logdir

    train(data_info,
            train_data_a, train_labels_a, test_data_a, test_labels_a,
            train_data_b, train_labels_b, test_data_b, test_labels_b,
            model_func=model_func,
            model_dir=model_dir,
            log_dir=log_dir,
            img_dir=args.imgdir,
            embedding_prefix=prefix,
            adaptation=adaptation,
            lr_multiplier=args.lrmult)
