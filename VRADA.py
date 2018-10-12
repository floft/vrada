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

# Make sure matplotlib is not interactive
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from plot import plot_embedding, plot_random_time_series, plot_real_time_series
from model import build_lstm, build_vrnn
from load_data import IteratorInitializerHook, \
    load_data, one_hot, \
    domain_labels, _get_input_fn, \
    load_data_sleep, load_data_mimiciii_ahrf, load_data_mimiciii_icd9

def compute_evaluation(sess,
    eval_input_hook_a, eval_input_hook_b,
    next_data_batch_test_a, next_labels_batch_test_a,
    next_data_batch_test_b, next_labels_batch_test_b,
    task_accuracy_sum, domain_accuracy_sum,
    source_domain, target_domain,
    x, y, task_classifier, domain, keep_prob, training,
    batch_size, auc_labels, auc_predictions, task_auc_all):
    """
    Run all the evaluation data to calculate accuracy and AUC

    We may not be able to fit all the evaluation data into memory, so we'll
    loop over it in batches and at the end calculate the accuracy.
    """
    # Evaluation set in batches (since it may not fit into memory
    # all at once)
    task_a = np.zeros((num_classes,), dtype=np.float32)
    task_b = np.zeros((num_classes,), dtype=np.float32)
    domain_a = 0
    domain_b = 0
    a_total = 0
    b_total = 0

    # For AUC, we need to keep track of all the predictions
    auc_a_labels = None
    auc_a_predictions = None
    auc_b_labels = None
    auc_b_predictions = None

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

            # If the number of evaluation examples is not divisible by the batch
            # size, then the last one will not be a full batch. Thus, we'll need
            # to pass in the proper domain labels with the correct length.
            if eval_data_a.shape[0] != batch_size or eval_data_b.shape[0] != batch_size:
                batch_source_domain = domain_labels(0, eval_data_a.shape[0])
                batch_target_domain = domain_labels(1, eval_data_b.shape[0])
            else:
                batch_source_domain = source_domain
                batch_target_domain = target_domain

            # Log summaries run on the evaluation/validation data
            batch_task_a, batch_domain_a, pred_a = sess.run(
                [task_accuracy_sum, domain_accuracy_sum, task_classifier], feed_dict={
                x: eval_data_a, y: eval_labels_a, domain: batch_source_domain,
                keep_prob: 1.0, training: False
            })
            batch_task_b, batch_domain_b, pred_b = sess.run(
                [task_accuracy_sum, domain_accuracy_sum, task_classifier], feed_dict={
                x: eval_data_b, y: eval_labels_b, domain: batch_target_domain,
                keep_prob: 1.0, training: False
            })

            # Update sums for computing accuracy
            task_a += batch_task_a
            task_b += batch_task_b
            domain_a += batch_domain_a
            domain_b += batch_domain_b
            a_total += eval_data_a.shape[0]
            b_total += eval_data_b.shape[0]

            # Save predictions for computing AUC
            if auc_a_labels is None:
                auc_a_labels = np.copy(eval_labels_a)
                auc_a_predictions = np.copy(pred_a)
                auc_b_labels = np.copy(eval_labels_b)
                auc_b_predictions = np.copy(pred_b)
            else:
                auc_a_labels = np.vstack([auc_a_labels, eval_labels_a])
                auc_a_predictions = np.vstack([auc_a_predictions, pred_a])
                auc_b_labels = np.vstack([auc_b_labels, eval_labels_b])
                auc_b_predictions = np.vstack([auc_b_predictions, pred_b])
        except tf.errors.OutOfRangeError:
            break

    task_a_accuracy = task_a / a_total
    domain_a_accuracy = domain_a / a_total
    task_b_accuracy = task_b / b_total
    domain_b_accuracy = domain_b / b_total

    # Computing AUC - we could do this out of TensorFlow, but I want to make
    # sure the calculation is exactly the same as the TensorFlow AUC function
    # we're using for the training batches
    task_a_auc = sess.run(task_auc_all, feed_dict={
        auc_labels: auc_a_labels, auc_predictions: auc_a_predictions
    })
    task_b_auc = sess.run(task_auc_all, feed_dict={
        auc_labels: auc_b_labels, auc_predictions: auc_b_predictions
    })

    return task_a_accuracy, domain_a_accuracy, \
        task_b_accuracy, domain_b_accuracy, \
        task_a_auc, task_b_auc

def evaluation_plots(sess,
    eval_input_hook_a, eval_input_hook_b,
    next_data_batch_test_a, next_labels_batch_test_a,
    next_data_batch_test_b, next_labels_batch_test_b,
    source_domain, target_domain,
    feature_extractor, x, keep_prob, training, adaptation,
    extra_model_outputs, num_features, first_step,
    tsne_filename=None,
    pca_filename=None,
    recon_a_filename=None,
    recon_b_filename=None,
    real_a_filename=None,
    real_b_filename=None):
    """
    Run the first batch of evaluation data through the feature extractor, then
    generate and return the PCA and t-SNE plots. Optionally, save these to a file
    as well.
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

    tsne_plot = plot_embedding(
        tsne, combined_labels.argmax(1), combined_domain.argmax(1),
        title=title + " - t-SNE", filename=tsne_filename)
    pca_plot = plot_embedding(pca, combined_labels.argmax(1), combined_domain.argmax(1),
        title=title + " - PCA", filename=pca_filename)

    plots = []
    plots.append(('tsne', tsne_plot))
    plots.append(('pca', pca_plot))

    # Output time-series "reconstructions" from our generator (if VRNN and we
    # only have a single-dimensional x, e.g. in the "trivial" datasets)
    if extra_model_outputs is not None:
        # We'll get the decoder's mu and sigma from the evaluation/validation set since
        # it's much larger than the training batches
        mu_a, sigma_a = sess.run(extra_model_outputs, feed_dict={
            x: eval_data_a, keep_prob: 1.0, training: False
        })

        mu_b, sigma_b = sess.run(extra_model_outputs, feed_dict={
            x: eval_data_b, keep_prob: 1.0, training: False
        })

        for i in range(num_features):
            recon_a_plot = plot_random_time_series(
                mu_a[:,:,i], sigma_a[:,:,i],
                title='VRNN Reconstruction (source domain, feature '+str(i)+')',
                filename=recon_a_filename)

            recon_b_plot = plot_random_time_series(
                mu_b[:,:,i], sigma_b[:,:,i],
                title='VRNN Reconstruction (target domain, feature '+str(i)+')',
                filename=recon_b_filename)

            plots.append(('feature_'+str(i)+'_reconstruction_a', recon_a_plot))
            plots.append(('feature_'+str(i)+'_reconstruction_b', recon_b_plot))

            # Real data -- but only plot once, since this doesn't change for the
            # evaluation data
            if first_step:
                real_a_plot = plot_real_time_series(
                    eval_data_a[:,:,i],
                    title='Real Data (source domain, feature '+str(i)+')',
                    filename=real_a_filename)

                real_b_plot = plot_real_time_series(
                    eval_data_b[:,:,i],
                    title='Real Data (target domain, feature '+str(i)+')',
                    filename=real_b_filename)

                plots.append(('feature_'+str(i)+'_real_a', real_a_plot))
                plots.append(('feature_'+str(i)+'_real_b', real_b_plot))

    return plots

def train(data_info,
        features_a, labels_a, test_features_a, test_labels_a,
        features_b, labels_b, test_features_b, test_labels_b,
        model_func=build_lstm,
        batch_size=128,
        num_steps=100000,
        learning_rate=0.0003,
        lr_multiplier=1,
        dropout_keep_prob=0.8,
        units=100,
        model_dir="models",
        log_dir="logs",
        model_save_steps=1000,
        log_save_steps=50,
        log_validation_accuracy_steps=250,
        log_extra_save_steps=1000,
        adaptation=True,
        multi_class=False):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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
            num_classes, num_features, adaptation, units, multi_class)

    # Accuracy of the classifiers -- https://stackoverflow.com/a/42608050/2698494
    #
    # Note: we also calculate the *sum* (not just the mean), since we need to
    # run these multiple times if we can't fit the entire validation set into
    # memory. Then afterwards we can divide by the size of the validation set.
    with tf.variable_scope("task_accuracy"):
        # If multi-class, then each output is a sigmoid independent of the others,
        # so for each class check >0.5 for predicting a "yes" for that class.
        if multi_class:
            equals = tf.cast(
                tf.equal(y, tf.cast(tf.greater(task_classifier, 0.5), tf.float32)),
                tf.float32)
        # If only predicting a single class (using softmax), then look for the
        # max value
        else:
            # e.g. [0.2 0.2 0.4 0.2] -> [0 0 1 0]
            pred_one_hot = tf.one_hot(tf.argmax(task_classifier, axis=-1), num_classes)
            # compare predicted one_hot and ground truth for each class
            equals = tf.cast(tf.equal(y, pred_one_hot), tf.float32)

        task_accuracy_sum = tf.reduce_sum(equals, axis=0)
        task_accuracy = tf.reduce_mean(equals, axis=0)
        task_accuracy_avg = tf.reduce_mean(task_accuracy)
    with tf.variable_scope("domain_accuracy"):
        equals = tf.cast(
            tf.equal(tf.argmax(domain, axis=-1), tf.argmax(domain_classifier, axis=-1)),
        tf.float32)
        domain_accuracy_sum = tf.reduce_sum(equals)
        domain_accuracy = tf.reduce_mean(equals)

    # Also compute AUC since that's what's given in some papers
    with tf.variable_scope("task_auc"):
        _, task_auc = tf.metrics.auc(labels=y, predictions=task_classifier)

        # Or, via placeholders if doing it in multiple batches
        auc_labels = tf.placeholder(tf.float32, [None, num_classes], name='labels')
        auc_predictions = tf.placeholder(tf.float32, [None, num_classes], name='predictions')
        _, task_auc_all = tf.metrics.auc(labels=auc_labels, predictions=auc_predictions)

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

    # Summaries - training and evaluation for both domains A and B
    training_a_summs = [
        tf.summary.scalar("loss/total_loss", total_loss),
        tf.summary.scalar("auc_task/source/training", task_auc),
        tf.summary.scalar("accuracy_task_avg/source/training", task_accuracy_avg),
        tf.summary.scalar("accuracy_domain/source/training", domain_accuracy),
    ]
    training_b_summs = [
        tf.summary.scalar("auc_task/target/training", task_auc),
        tf.summary.scalar("accuracy_task_avg/target/training", task_accuracy_avg),
        tf.summary.scalar("accuracy_domain/target/training", domain_accuracy)
    ]
    with tf.variable_scope("task_accuracy", auxiliary_name_scope=False):
        for i in range(num_classes):
            class_acc = tf.reshape(tf.slice(task_accuracy, [i], [1]), [])

            training_a_summs += [
                tf.summary.scalar("accuracy_task_class%d/source/training" % i, class_acc),
            ]
            training_b_summs += [
                tf.summary.scalar("accuracy_task_class%d/target/training" % i, class_acc),
            ]
    training_summaries_a = tf.summary.merge(training_a_summs)
    training_summaries_extra_a = tf.summary.merge(model_summaries)
    training_summaries_b = tf.summary.merge(training_b_summs)

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

                # Train everything in one step and domain more next. This seemed
                # to work better for me than just nondomain then domain, though
                # it seems likely the results would be similar.
                sess.run(train_all, feed_dict={
                    x: combined_x, y: combined_labels, domain: combined_domain,
                    grl_lambda: grl_lambda_value,
                    keep_prob: dropout_keep_prob, lr: lr_value, training: True
                })

                # Update domain more
                #
                # Depending on the num_steps, your learning rate, etc. it may be
                # beneficial to have a different learning rate here -- hence the
                # lr_multiplier option. This may also depend on your dataset though.
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

            # Log validation accuracy/AUC less frequently
            if i%log_validation_accuracy_steps == 0:
                # Evaluation accuracy and AUC
                task_a_accuracy, domain_a_accuracy, \
                task_b_accuracy, domain_b_accuracy, \
                task_a_auc, task_b_auc = compute_evaluation(sess,
                    eval_input_hook_a, eval_input_hook_b,
                    next_data_batch_test_a, next_labels_batch_test_a,
                    next_data_batch_test_b, next_labels_batch_test_b,
                    task_accuracy_sum, domain_accuracy_sum,
                    source_domain, target_domain,
                    x, y, task_classifier, domain, keep_prob, training,
                    batch_size, auc_labels, auc_predictions, task_auc_all)

                task_source_val = []
                for i in range(num_classes):
                    task_source_val += [tf.Summary(value=[tf.Summary.Value(
                        tag="accuracy_task_class%d/source/validation" % i,
                        simple_value=task_a_accuracy[i]
                    )])]
                task_source_val_avg = tf.Summary(value=[tf.Summary.Value(
                        tag="accuracy_task_avg/source/validation",
                        simple_value=np.mean(task_a_accuracy)
                    )])
                domain_source_val = tf.Summary(value=[tf.Summary.Value(
                    tag="accuracy_domain/source/validation",
                    simple_value=domain_a_accuracy
                )])
                task_source_auc_val = tf.Summary(value=[tf.Summary.Value(
                    tag="auc_task/source/validation",
                    simple_value=task_a_auc
                )])
                task_target_val = []
                for i in range(num_classes):
                    task_target_val += [tf.Summary(value=[tf.Summary.Value(
                        tag="accuracy_task_class%d/target/validation" % i,
                        simple_value=task_b_accuracy[i]
                    )])]
                task_target_val_avg = tf.Summary(value=[tf.Summary.Value(
                        tag="accuracy_task_avg/target/validation",
                        simple_value=np.mean(task_b_accuracy)
                    )])
                domain_target_val = tf.Summary(value=[tf.Summary.Value(
                    tag="accuracy_domain/target/validation",
                    simple_value=domain_b_accuracy
                )])
                task_target_auc_val = tf.Summary(value=[tf.Summary.Value(
                    tag="auc_task/target/validation",
                    simple_value=task_b_auc
                )])
                for s in task_source_val:
                    writer.add_summary(s, step)
                writer.add_summary(task_source_val_avg, step)
                writer.add_summary(domain_source_val, step)
                writer.add_summary(task_source_auc_val, step)
                for s in task_target_val:
                    writer.add_summary(s, step)
                writer.add_summary(task_target_val_avg, step)
                writer.add_summary(domain_target_val, step)
                writer.add_summary(task_target_auc_val, step)

            # Larger stuff like weights and t-SNE plots occasionally
            if i%log_extra_save_steps == 0:
                # Training weights
                summ = sess.run(training_summaries_extra_a, feed_dict={
                    x: data_batch_a, y: labels_batch_a, domain: source_domain,
                    keep_prob: 1.0, training: False
                })
                writer.add_summary(summ, step)

                # t-SNE, PCA, and VRNN reconstruction plots
                first_step = i==0 # only plot real ones once
                plots = evaluation_plots(sess,
                    eval_input_hook_a, eval_input_hook_b,
                    next_data_batch_test_a, next_labels_batch_test_a,
                    next_data_batch_test_b, next_labels_batch_test_b,
                    source_domain, target_domain,
                    feature_extractor, x, keep_prob, training, adaptation,
                    extra_model_outputs, num_features, first_step)

                for name, buf in plots:
                    # See: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
                    plot = tf.Summary.Image(encoded_image_string=buf)
                    summ = tf.Summary(value=[tf.Summary.Value(
                        tag=name, image=plot)])
                    writer.add_summary(summ, step)

                # Make sure we write to disk before too long so we can monitor live in
                # TensorBoard. If it's too delayed we won't be able to detect problems
                # for a long time.
                writer.flush()

        writer.flush()

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
    parser.add_argument('--lstm', dest='lstm', action='store_true',
        help="Use LSTM model")
    parser.add_argument('--no-lstm', dest='lstm', action='store_false',
        help="Do not LSTM model (default)")
    parser.add_argument('--vrnn', dest='vrnn', action='store_true',
        help="Use VRNN model")
    parser.add_argument('--no-vrnn', dest='vrnn', action='store_false',
        help="Do not use VRNN model (default)")
    parser.add_argument('--lstm-da', dest='lstm_da', action='store_true',
        help="Use LSTM-DA model")
    parser.add_argument('--no-lstm-da', dest='lstm_da', action='store_false',
        help="Do not use LSTM-DA model (default)")
    parser.add_argument('--vrnn-da', dest='vrnn_da', action='store_true',
        help="Use VRNN-DA model")
    parser.add_argument('--no-vrnn-da', dest='vrnn_da', action='store_false',
        help="Do not use VRNN-DA model (default)")
    parser.add_argument('--mimic-icd9', dest='mimic_icd9', action='store_true',
        help="Run on the MIMIC-III ICD-9 code prediction dataset")
    parser.add_argument('--no-mimic-icd9', dest='mimic_icd9', action='store_false',
        help="Do not run on the MIMIC-III ICD-9 code prediction dataset (default)")
    parser.add_argument('--mimic-ahrf', dest='mimic_ahrf', action='store_true',
        help="Run on the MIMIC-III Adult AHRF dataset (warning: not just AHRF)")
    parser.add_argument('--no-mimic-ahrf', dest='mimic_ahrf', action='store_false',
        help="Do not run on the MIMIC-III Adult AHRF dataset (default)")
    parser.add_argument('--sleep', dest='sleep', action='store_true',
        help="Run on the RF sleep stage dataset")
    parser.add_argument('--no-sleep', dest='sleep', action='store_false',
        help="Do not run on the RF sleep stage dataset (default)")
    parser.add_argument('--trivial-line', dest='trivial_line', action='store_true',
        help="Run on the trivial synthetic line dataset")
    parser.add_argument('--no-trivial-line', dest='trivial_line', action='store_false',
        help="Do not run on the trivial synthetic line dataset (default)")
    parser.add_argument('--trivial-sine', dest='trivial_sine', action='store_true',
        help="Run on the trivial synthetic sine dataset")
    parser.add_argument('--no-trivial-sine', dest='trivial_sine', action='store_false',
        help="Do not run on the trivial synthetic sine dataset (default)")
    parser.add_argument('--units', default=100, type=int,
        help="Number of LSTM hidden units and VRNN latent variable size (default 100)")
    parser.add_argument('--steps', default=100000, type=int,
        help="Number of training steps to run (default 100000)")
    parser.add_argument('--batch', default=128, type=int,
        help="Batch size to use (default 128, decrease if you run out of memory)")
    parser.add_argument('--lr', default=0.0003, type=float,
        help="Learning rate for training (default 0.0003)")
    parser.add_argument('--lr-mult', default=1.0, type=float,
        help="Multiplier for extra discriminator training learning rate (default 1)")
    parser.add_argument('--dropout', default=0.8, type=float,
        help="Keep probability for dropout (default 0.8)")
    parser.add_argument('--model-steps', default=1000, type=int,
        help="Save the model every so many steps (default 1000)")
    parser.add_argument('--log-steps', default=50, type=int,
        help="Log training losses and accuracy every so many steps (default 50)")
    parser.add_argument('--log-steps-val', default=250, type=int,
        help="Log validation accuracy and AUC every so many steps (default 250)")
    parser.add_argument('--log-steps-slow', default=1000, type=int,
        help="Log weights, plots, etc. every so many steps (default 1000)")
    parser.add_argument('--debug', dest='debug', action='store_true',
        help="Start new log/model/images rather than continuing from previous run")
    parser.add_argument('--debug-num', default=-1, type=int,
        help="Specify exact log/model/images number to use rather than incrementing from last. " \
            +"(Don't pass both this and --debug at the same time.)")
    parser.set_defaults(
        lstm=False, vrnn=False, lstm_da=False, vrnn_da=False,
        mimic_ahrf=False, mimic_icd9=False, sleep=False, trivial_line=False,
        trivial_sine=False, debug=False)
    args = parser.parse_args()

    # Load datasets - domains A & B
    assert args.mimic_ahrf + args.mimic_icd9 + \
        + args.sleep \
        + args.trivial_line + args.trivial_sine == 1, \
        "Must specify exactly one dataset to use"

    if args.trivial_line:
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
        multi_class = False # Predict only one class
    elif args.trivial_sine:
        # Change in y-intercept
        train_data_a, train_labels_a = load_data("datasets/trivial/positive_sine_TRAIN")
        test_data_a, test_labels_a = load_data("datasets/trivial/positive_sine_TEST")
        train_data_b, train_labels_b = load_data("datasets/trivial/positive_sine_low_TRAIN")
        test_data_b, test_labels_b = load_data("datasets/trivial/positive_sine_low_TEST")

        # Information about dataset - same for both domains on these datasets
        index_one = True # Labels start from 1 not 0
        num_features = 1
        time_steps = train_data_a.shape[1]
        num_classes = len(np.unique(train_labels_a))
        data_info = (time_steps, num_features, num_classes)
        multi_class = False # Predict only one class
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
        multi_class = False # Predict only one class
    elif args.mimic_ahrf:
        train_data_a, train_labels_a, \
        test_data_a, test_labels_a, \
        train_data_b, train_labels_b, \
        test_data_b, test_labels_b = load_data_mimiciii_ahrf()

        # Information about dataset
        index_one = False # Labels start from 0
        num_features = train_data_a.shape[2]
        time_steps = train_data_a.shape[1]
        num_classes = len(np.unique(train_labels_a))
        assert num_classes == 2, "Should be 2 classes (binary) for MIMIC-III AHRF"
        data_info = (time_steps, num_features, num_classes)
        multi_class = False # Predict only one class
    else: # args.mimic_icd9
        train_data_a, train_labels_a, \
        test_data_a, test_labels_a, \
        train_data_b, train_labels_b, \
        test_data_b, test_labels_b = load_data_mimiciii_icd9()

        # Information about dataset
        index_one = False # Labels start from 0
        num_features = train_data_a.shape[2]
        time_steps = train_data_a.shape[1]
        num_classes = train_labels_a.shape[1]
        assert num_classes == 20, "Should be 20 ICD-9 categories"
        data_info = (time_steps, num_features, num_classes)
        multi_class = True # Predict any number of the classes at once

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

    # Use the number specified on the command line (higher precidence than --debug)
    if args.debug_num >= 0:
        attempt = args.debug_num
        print("Debugging attempt:", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(args.modeldir, prefix)
        log_dir = os.path.join(args.logdir, prefix)
    # Find last one, increment number
    elif args.debug:
        attempt = last_modified_number(args.logdir, prefix+"*")
        attempt = attempt+1 if attempt is not None else 1
        print("Debugging attempt:", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(args.modeldir, prefix)
        log_dir = os.path.join(args.logdir, prefix)
    # If no debugging modes, use the model and log directory as-is
    else:
        model_dir = args.modeldir
        log_dir = args.logdir

    train(data_info,
            train_data_a, train_labels_a, test_data_a, test_labels_a,
            train_data_b, train_labels_b, test_data_b, test_labels_b,
            model_func=model_func,
            model_dir=model_dir,
            log_dir=log_dir,
            adaptation=adaptation,
            num_steps=args.steps,
            learning_rate=args.lr,
            lr_multiplier=args.lr_mult,
            units=args.units,
            batch_size=args.batch,
            dropout_keep_prob=args.dropout,
            model_save_steps=args.model_steps,
            log_save_steps=args.log_steps,
            log_validation_accuracy_steps=args.log_steps_val,
            log_extra_save_steps=args.log_steps_slow,
            multi_class=multi_class)
