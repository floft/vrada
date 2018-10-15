"""
Create models

This provides the functions like build_lstm and build_vrnn that are used in training.
"""
import tensorflow as tf
layers = tf.contrib.layers
framework = tf.contrib.framework

from VRNN import VRNNCell
from flip_gradient import flip_gradient

def build_rnn(x, keep_prob, layers):
    """
    Multi-layer LSTM
    https://github.com/GarrettHoffman/lstm-oreilly

    x, keep_prob - placeholders
    layers - cell for each layer, e.g. [LSTMCell(...), LSTMCell(...), ...]
    """

    #drops = [tf.contrib.rnn.DropoutWrapper(l, output_keep_prob=keep_prob) for l in layers]
    #cell = tf.contrib.rnn.MultiRNNCell(drops)
    #cell = tf.contrib.rnn.MultiRNNCell(layers)
    cell = layers[0] # We won't use multiple layers at the moment

    batch_size = tf.shape(x)[0]
    initial_state = cell.zero_state(batch_size, tf.float32)

    # TODO try tf.nn.bidirectional_dynamic_rnn
    outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)

    return initial_state, outputs, cell, final_state

def classifier(x, num_classes, keep_prob, training, batch_norm):
    """
    We'll use the same clasifier for task or domain classification

    Same as was used in the VRADA paper (see paper appendix)

    Returns both output without applying softmax for use in loss function
    and after for use in prediction. See softmax_cross_entropy_with_logits_v2
    documentation: "This op expects unscaled logits, ..."

    Also returns sigmoid output for if doing multi-class classification.
    """
    classifier_output = x
    num_layers = 4

    for i in range(num_layers):
        with tf.variable_scope("layer_"+str(i)):
            # Last layer has desired output size, otherwise use a fixed size
            if i == num_layers-1:
                num_features = num_classes
            else:
                num_features = 50

            classifier_output = tf.contrib.layers.fully_connected(
                    classifier_output, num_features, activation_fn=None)
            classifier_output = tf.nn.dropout(classifier_output, keep_prob)

            if batch_norm:
                classifier_output = tf.layers.batch_normalization(
                    classifier_output, training=training)

            # Last activation is softmax, which we will apply afterwards
            if i != num_layers-1:
                classifier_output = tf.nn.relu(classifier_output)

    sigmoid_output = tf.nn.sigmoid(classifier_output)
    softmax_output = tf.nn.softmax(classifier_output)

    return classifier_output, softmax_output, sigmoid_output

def build_model(x, y, domain, grl_lambda, keep_prob, training,
        num_classes, adaptation=True, multi_class=False, class_weights=1.0,
        batch_norm=False, two_domain_classifiers=False, log_outputs=True):
    """
    Creates the feature extractor, task classifier, domain classifier

    Inputs:
        x -- fed into feature extractor
        y -- task labels
        domain -- [[1,0], [0,1], ...] for source or target domain
        glr_lambda -- float placeholder for lambda for gradient reversal layer
        keep_prob -- float placeholder for dropout probability
        training -- boolean placeholder for if we're training
        adaptation -- boolean whether we wish to perform adaptation or not
        multi_class -- boolean whether to use sigmoid (for multi-class) or softmax
        batch_norm -- boolean whether to use BatchNorm
        two_domain_classifiers -- an experiment, not recommended to use
        log_outputs -- boolean whether we want to log outputs to for TensorBoard
        class_weights -- weights for handling large class imbalances (probably
            pass in [class0_weight, class1_weight, ... classN_weight])
    Outputs:
        task_output, domain_softmax -- predictions of classifiers
        task_loss, domain_loss -- losses
        feature_extractor -- output of feature extractor (e.g. for t-SNE)
        summaries -- more summaries to save
    """

    with tf.variable_scope("feature_extractor"):
        feature_extractor = x
        num_layers = 3

        for i in range(num_layers):
            with tf.variable_scope("layer_"+str(i)):
                feature_extractor = tf.contrib.layers.fully_connected(
                    feature_extractor, 100, activation_fn=None)
                feature_extractor = tf.nn.dropout(feature_extractor, keep_prob)

                if batch_norm:
                    feature_extractor = tf.layers.batch_normalization(
                        feature_extractor, training=training)

                feature_extractor = tf.nn.relu(feature_extractor)

    # Pass last output to fully connected then softmax to get class prediction
    with tf.variable_scope("task_classifier"):
        task_classifier, task_softmax, task_sigmoid = classifier(
            feature_extractor, num_classes, keep_prob, training, batch_norm)

    # Also pass output to domain classifier
    # Note: always have 2 domains, so set outputs to 2
    with tf.variable_scope("domain_classifier"):
        gradient_reversal_layer = flip_gradient(feature_extractor, grl_lambda)
        domain_classifier, domain_softmax, _ = classifier(
            gradient_reversal_layer, 2, keep_prob, training, batch_norm)

    # Maybe try one before the feature extractor too
    if two_domain_classifiers:
        with tf.variable_scope("domain_classifier2"):
            gradient_reversal_layer2 = flip_gradient(x, grl_lambda)
            domain_classifier2, _, _ = classifier(
                gradient_reversal_layer2, 2, keep_prob, training, batch_norm)

    # If doing domain adaptation, then we'll need to ignore the second half of the
    # batch for task classification during training since we don't know the labels
    # of the target data
    if adaptation:
        with tf.variable_scope("only_use_source_labels"):
            # Note: this is twice the batch_size in the train() function since we cut
            # it in half there -- this is the sum of both source and target data
            batch_size = tf.shape(feature_extractor)[0]

            # Note: I'm doing this after the classification layers because if you do
            # it before, then fully_connected complains that the last dimension is
            # None (i.e. not known till we run the graph). Thus, we'll do it after
            # all the fully-connected layers.
            #
            # Alternatively, I could do matmul(weights, task_input) + bias and store
            # weights on my own if I do really need to do this at some point.
            #
            # See: https://github.com/pumpikano/tf-dann/blob/master/Blobs-DANN.ipynb
            task_classifier = tf.cond(training,
                lambda: tf.slice(task_classifier, [0, 0], [batch_size // 2, -1]),
                lambda: task_classifier)
            task_softmax = tf.cond(training,
                lambda: tf.slice(task_softmax, [0, 0], [batch_size // 2, -1]),
                lambda: task_softmax)
            task_sigmoid = tf.cond(training,
                lambda: tf.slice(task_sigmoid, [0, 0], [batch_size // 2, -1]),
                lambda: task_sigmoid)
            y = tf.cond(training,
                lambda: tf.slice(y, [0, 0], [batch_size // 2, -1]),
                lambda: y)

    # Losses
    with tf.variable_scope("task_loss"):
        # Tile the class weights to match the batch size
        #
        # e.g., if the weights are [1,2,3,4] and we have a batch of size 2, we get:
        #  [[1,2,3,4],
        #   [1,2,3,4]]
        if not isinstance(class_weights, float) and not isinstance(class_weights, int):
            class_weights_reshape = tf.reshape(class_weights,
                [1,tf.shape(class_weights)[0]])
            tiled_class_weights = tf.tile(class_weights_reshape,
                [tf.shape(y)[0],1])

            # If not multi-class, then there needs to be one weight for each
            # item in the batch based on which class that item was predicted to
            # be
            #
            # e.g. if we predicted classes [[0,1],[1,0],[1,0]] (i.e. class 1,
            # class 0, class 0) for a batch size of two, and we have weights
            # [2,3] we should output: [3,2,2] for the weights for this batch
            which_label = tf.argmax(task_classifier, axis=-1) # e.g. [1,0,0] for above
            # Then, get the weights based on which class each was
            batch_class_weights = tf.gather(class_weights, which_label)
        # If it's just the default 1.0 or some scalar, then don't bother
        # expanding to match the batch size
        else:
            tiled_class_weights = class_weights
            batch_class_weights = class_weights

        # If multi-class (i.e. predict any number of the classes not necessarily
        # just one), use a different TensorFlow loss function that treats each
        # output separately (not doing softmax, where we care about the max one)
        if multi_class:
            task_loss = tf.losses.sigmoid_cross_entropy(
                y, task_classifier, tiled_class_weights)
        else:
            task_loss = tf.losses.softmax_cross_entropy(
                y, task_classifier, batch_class_weights)

    with tf.variable_scope("domain_loss"):
        domain_loss = tf.losses.softmax_cross_entropy(domain, domain_classifier)

        if two_domain_classifiers:
            domain_loss += tf.losses.softmax_cross_entropy(
                domain, domain_classifier2)

    # If multi-class the task output will be sigmoid rather than softmax
    if multi_class:
        task_output = task_sigmoid
    else:
        task_output = task_softmax

    # Extra summaries
    summaries = [
        tf.summary.scalar("loss/task_loss", task_loss),
        tf.summary.scalar("loss/domain_loss", domain_loss),
    ]

    if log_outputs:
        summaries += [
            tf.summary.histogram("outputs/feature_extractor", feature_extractor),
            tf.summary.histogram("outputs/domain_classifier", domain_softmax),
        ]

        with tf.variable_scope("outputs"):
            for i in range(num_classes):
                summaries += [
                    tf.summary.histogram("task_classifier_%d" % i,
                        tf.slice(task_output, [0,i], [tf.shape(task_output)[0],1]))
                ]

    return task_output, domain_softmax, task_loss, domain_loss, \
        feature_extractor, summaries

def build_lstm(x, y, domain, grl_lambda, keep_prob, training,
            num_classes, num_features, adaptation, units,
            multi_class=False, class_weights=1.0):
    """ LSTM for a baseline """
    # Build LSTM
    with tf.variable_scope("rnn_model"):
        _, outputs, _, _ = build_rnn(x, keep_prob, [
            tf.contrib.rnn.BasicLSTMCell(units),
            #tf.contrib.rnn.LayerNormBasicLSTMCell(100, dropout_keep_prob=keep_prob),
        ])

        rnn_output = outputs[:, -1]

    # Other model components passing in output from RNN
    task_output, domain_softmax, task_loss, domain_loss, \
        feature_extractor, summaries = build_model(
            rnn_output, y, domain, grl_lambda, keep_prob, training,
            num_classes, adaptation, multi_class, class_weights)

    # Total loss is the sum
    with tf.variable_scope("total_loss"):
        total_loss = task_loss

        if adaptation:
            total_loss += domain_loss

    # We can't generate with an LSTM
    extra_outputs = None

    return task_output, domain_softmax, total_loss, \
        feature_extractor, summaries, extra_outputs

def build_vrnn(x, y, domain, grl_lambda, keep_prob, training,
            num_classes, num_features, adaptation, units,
            multi_class=False, class_weights=1.0,
            eps=1e-9, use_z=True,
            log_outputs=False, log_weights=False):
    """ VRNN model """
    # Build VRNN
    with tf.variable_scope("rnn_model"):
        _, outputs, _, _ = build_rnn(x, keep_prob, [
            VRNNCell(num_features, units, units, training, batch_norm=False),
        ])
        # Note: if you try using more than one layer above, then you need to
        # change the loss since for instance if you put an LSTM layer before
        # the VRNN cell, then no longer is the input to the layer x as
        # specified in the loss but now it's the output of the first LSTM layer
        # that the VRNN layer should be learning how to reconstruct. Thus, for
        # now I'll keep it simple and not have multiple layers.

        h, c, \
        encoder_mu, encoder_sigma, \
        decoder_mu, decoder_sigma, \
        prior_mu, prior_sigma, \
        x_1, z_1, \
            = outputs

        # VRADA uses z not h
        if use_z:
            rnn_output = z_1[:,-1]
        else:
            rnn_output = h[:,-1]

    # Other model components passing in output from RNN
    task_output, domain_softmax, task_loss, domain_loss, \
        feature_extractor, summaries = build_model(
            rnn_output, y, domain, grl_lambda, keep_prob, training,
            num_classes, adaptation, multi_class, class_weights)

    # Loss
    #
    # KL divergence
    # https://stats.stackexchange.com/q/7440
    # https://github.com/kimkilho/tensorflow-vrnn/blob/master/main.py
    with tf.variable_scope("kl_gaussian"):
        kl_loss = tf.reduce_mean(tf.reduce_mean(
                tf.log(tf.maximum(eps, prior_sigma)) - tf.log(tf.maximum(eps, encoder_sigma))
                + 0.5*(tf.square(encoder_sigma) + tf.square(encoder_mu - prior_mu))
                    / tf.maximum(eps, tf.square(prior_sigma))
                - 0.5,
            axis=1), axis=1)

    # Reshape [batch_size,time_steps,num_features] -> [batch_size*time_steps,num_features]
    # so that (decoder_mu - x) will broadcast correctly
    #x_transpose = tf.transpose(x, [0, 2, 1])
    #decoder_mu_reshape = tf.reshape(decoder_mu, [tf.shape(decoder_mu)[0]*tf.shape(decoder_mu)[1], tf.shape(decoder_mu)[2]])
    #x_reshape = tf.reshape(x, [tf.shape(x)[0]*tf.shape(x)[1], tf.shape(x)[2]])

    # Negative log likelihood:
    # https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf
    # https://fairyonice.github.io/Create-a-neural-net-with-a-negative-log-likelihood-as-a-loss.html
    with tf.variable_scope("negative_log_likelihood"):
        #likelihood_loss = tf.reduce_sum(tf.squared_difference(x, x_1), 1)
        likelihood_loss = 0.5*tf.reduce_mean(tf.reduce_mean(
            tf.square(decoder_mu - x) / tf.maximum(eps, tf.square(decoder_sigma))
            + tf.log(tf.maximum(eps, tf.square(decoder_sigma))),
        axis=1), axis=1)

    # Total loss is sum of all of them
    with tf.variable_scope("total_loss"):
        total_loss = task_loss + tf.reduce_mean(kl_loss) + tf.reduce_mean(likelihood_loss)

        if adaptation:
            total_loss += domain_loss

    # Extra summaries
    summaries += [
        tf.summary.scalar("loss/kl", tf.reduce_mean(kl_loss)),
        tf.summary.scalar("loss/likelihood", tf.reduce_mean(likelihood_loss)),
    ]

    if log_outputs:
        summaries += [
            tf.summary.histogram("outputs/phi_x", x_1),
            tf.summary.histogram("outputs/phi_z", z_1),
        ]

    if log_weights:
        summaries += [
            tf.summary.histogram("encoder/mu", encoder_mu),
            tf.summary.histogram("encoder/sigma", encoder_sigma),
            tf.summary.histogram("decoder/mu", decoder_mu),
            tf.summary.histogram("decoder/sigma", decoder_sigma),
            tf.summary.histogram("prior/mu", prior_mu),
            tf.summary.histogram("prior/sigma", prior_sigma),
        ]

    # So we can generate sample time-series as well
    extra_outputs = [
        decoder_mu, decoder_sigma,
    ]

    return task_output, domain_softmax, total_loss, \
        feature_extractor, summaries, extra_outputs

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.3)

def cnn(x, keep_prob):
    """ Simple CNN, taken from tensorflow-experiments/VAE """
    with framework.arg_scope([layers.conv2d], num_outputs=64, kernel_size=4,
                    padding='same', activation_fn=leaky_relu):
        n  = layers.conv2d(x, stride=2)
        n  = tf.nn.dropout(n, keep_prob)
        n  = layers.conv2d(n, stride=2)
        n  = tf.nn.dropout(n, keep_prob)
        n  = layers.conv2d(n, stride=1)
        n  = tf.nn.dropout(n, keep_prob)
        n  = layers.flatten(n)

    return n

def build_cnn(x, y, domain, grl_lambda, keep_prob, training,
            num_classes, num_features, adaptation, units,
            multi_class=False, class_weights=1.0):
    """ CNN for image data rather than time-series data """
    # Build CNN
    with tf.variable_scope("cnn_model"):
        cnn_output = cnn(x, keep_prob)

    # Other model components passing in output from CNN
    task_output, domain_softmax, task_loss, domain_loss, \
        feature_extractor, summaries = build_model(
            cnn_output, y, domain, grl_lambda, keep_prob, training,
            num_classes, adaptation, multi_class, class_weights)

    # Total loss is the sum
    with tf.variable_scope("total_loss"):
        total_loss = task_loss

        if adaptation:
            total_loss += domain_loss

    # We can't generate with this CNN
    extra_outputs = None

    return task_output, domain_softmax, total_loss, \
        feature_extractor, summaries, extra_outputs
