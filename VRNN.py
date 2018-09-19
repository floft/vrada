"""
VRNN implementation

Based on:
 - https://github.com/phreeza/tensorflow-vrnn/blob/master/model_vrnn.py
 - https://github.com/kimkilho/tensorflow-vrnn/blob/master/cell.py
 - https://github.com/kimkilho/tensorflow-vrnn/blob/master/main.py
 - https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/ops/rnn_cell_impl.py
"""
import tensorflow as tf

class VRNNCell(tf.contrib.rnn.LayerRNNCell):
    def __init__(self, x_dim, h_dim, z_dim, training, batch_norm=False, **kwargs):
        self.batch_norm = batch_norm
        self.training = training # placeholder for batch norm

        # Dimensions of x input, hidden layers, latent variable (z)
        self.n_x = x_dim
        self.n_h = h_dim
        self.n_z = z_dim

        # Dimensions of phi(z)
        self.n_x_1 = x_dim
        self.n_z_1 = z_dim

        # Dimensions of encoder, decoder, and prior
        self.n_enc_hidden = z_dim
        self.n_dec_hidden = x_dim
        self.n_prior_hidden = z_dim

        # What cell we're going to use internally for the RNN
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.n_h)
             #input_shape=(None, self.n_dec_hidden+self.n_z_1))

        super(VRNNCell, self).__init__(**kwargs)

    @property
    def state_size(self):
        # Note: first two are the state of the LSTM
        return (self.n_h, self.n_h,
                self.n_z, self.n_z,
                self.n_x, self.n_x,
                self.n_z, self.n_z,
                self.n_x_1, self.n_z_1)

    @property
    def output_size(self):
        """ LSTM output is h which is in [0] """
        return (self.n_h, self.n_h,
                self.n_z, self.n_z,
                self.n_x, self.n_x,
                self.n_z, self.n_z,
                self.n_x_1, self.n_z_1)

    def build(self, input_shape):
        # TODO grab input size from input_shape rather than passing in
        # num_features to the constructor

        # Input: previous hidden state
        self.prior_h = self.add_variable('prior/hidden/weights',
            shape=(self.n_h, self.n_prior_hidden), initializer=tf.glorot_uniform_initializer())
        self.prior_mu = self.add_variable('prior/mu/weights',
            shape=(self.n_prior_hidden, self.n_z), initializer=tf.glorot_uniform_initializer())
        self.prior_sigma = self.add_variable('prior/sigma/weights',
            shape=(self.n_prior_hidden, self.n_z), initializer=tf.glorot_uniform_initializer())

        if not self.batch_norm:
            self.prior_h_b = self.add_variable('prior/hidden/bias',
                shape=(self.n_prior_hidden,), initializer=tf.constant_initializer())
            self.prior_sigma_b = self.add_variable('prior/sigma/bias',
                shape=(self.n_z,), initializer=tf.constant_initializer())
        self.prior_mu_b = self.add_variable('prior/mu/bias',
            shape=(self.n_z,), initializer=tf.constant_initializer())

        # Input: x
        self.x_1 = self.add_variable('phi_x/weights',
            shape=(self.n_x, self.n_x_1), initializer=tf.glorot_uniform_initializer())

        if not self.batch_norm:
            self.x_1_b = self.add_variable('phi_x/bias',
                shape=(self.n_x_1,), initializer=tf.constant_initializer())

        # Input: x and previous hidden state
        self.encoder_h = self.add_variable('encoder/hidden/weights',
            shape=(self.n_x_1+self.n_h, self.n_enc_hidden), initializer=tf.glorot_uniform_initializer())
        self.encoder_mu = self.add_variable('encoder/mu/weights',
            shape=(self.n_enc_hidden, self.n_z), initializer=tf.glorot_uniform_initializer())
        self.encoder_sigma = self.add_variable('encoder/sigma/weights',
            shape=(self.n_enc_hidden, self.n_z), initializer=tf.glorot_uniform_initializer())

        if not self.batch_norm:
            self.encoder_h_b = self.add_variable('encoder/hidden/bias',
                shape=(self.n_enc_hidden,), initializer=tf.constant_initializer())
            self.encoder_sigma_b = self.add_variable('encoder/sigma/bias',
                shape=(self.n_z,), initializer=tf.constant_initializer())
        self.encoder_mu_b = self.add_variable('encoder/mu/bias',
            shape=(self.n_z,), initializer=tf.constant_initializer())

        # Input: z = enc_sigma*eps + enc_mu -- i.e. reparameterization trick
        self.z_1 = self.add_variable('phi_z/weights',
            shape=(self.n_z, self.n_z_1), initializer=tf.glorot_uniform_initializer())

        if not self.batch_norm:
            self.z_1_b = self.add_variable('phi_z/bias',
                shape=(self.n_z_1,), initializer=tf.constant_initializer())

        # Input: latent variable (z) and previous hidden state
        self.decoder_h = self.add_variable('decoder/hidden/weights',
            shape=(self.n_z+self.n_h, self.n_dec_hidden), initializer=tf.glorot_uniform_initializer())
        self.decoder_mu = self.add_variable('decoder/mu/weights',
            shape=(self.n_dec_hidden, self.n_x), initializer=tf.glorot_uniform_initializer())
        self.decoder_sigma = self.add_variable('decoder/sigma/weights',
            shape=(self.n_dec_hidden, self.n_x), initializer=tf.glorot_uniform_initializer())

        if not self.batch_norm:
            self.decoder_h_b = self.add_variable('decoder/hidden/bias',
                shape=(self.n_dec_hidden,), initializer=tf.constant_initializer())
            self.decoder_sigma_b = self.add_variable('decoder/sigma/bias',
                shape=(self.n_x,), initializer=tf.constant_initializer())
        self.decoder_mu_b = self.add_variable('decoder/mu/bias',
            shape=(self.n_x,), initializer=tf.constant_initializer())

        super(VRNNCell, self).build(input_shape)

    def call(self, inputs, states):
        # Get relevant states
        h = states[0]
        c = states[1] # only passed to the LSTM

        # Input: previous hidden state (h)
        #
        # Note: update_collections=None from https://github.com/tensorflow/tensorflow/issues/6087
        # And, that's why I'm not using tf.layers.batch_normalization
        if self.batch_norm:
            prior_h = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(h, self.prior_h), is_training=self.training, updates_collections=None))
            prior_sigma = tf.nn.softplus(tf.contrib.layers.batch_norm(tf.matmul(prior_h, self.prior_sigma), is_training=self.training, updates_collections=None)) # >= 0
        else:
            prior_h = tf.nn.relu(tf.matmul(h, self.prior_h) + self.prior_h_b)
            prior_sigma = tf.nn.softplus(tf.matmul(prior_h, self.prior_sigma) + self.prior_sigma_b) # >= 0
        prior_mu = tf.matmul(prior_h, self.prior_mu) + self.prior_mu_b

        # Input: x
        #
        # Note: removed ReLU since in the dataset not all x values are positive
        if self.batch_norm:
            x_1 = tf.contrib.layers.batch_norm(tf.matmul(inputs, self.x_1), is_training=self.training, updates_collections=None)
        else:
            x_1 = tf.matmul(inputs, self.x_1) + self.x_1_b

        # Input: x and previous hidden state
        encoder_input = tf.concat((x_1, h), 1)
        if self.batch_norm:
            encoder_h = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(encoder_input, self.encoder_h), is_training=self.training, updates_collections=None))
            encoder_sigma = tf.nn.softplus(tf.contrib.layers.batch_norm(tf.matmul(encoder_h, self.encoder_sigma), is_training=self.training, updates_collections=None))
        else:
            encoder_h = tf.nn.relu(tf.matmul(encoder_input, self.encoder_h) + self.encoder_h_b)
            encoder_sigma = tf.nn.softplus(tf.matmul(encoder_h, self.encoder_sigma) + self.encoder_sigma_b)
        encoder_mu = tf.matmul(encoder_h, self.encoder_mu) + self.encoder_mu_b

        # Input: z = enc_sigma*eps + enc_mu -- i.e. reparameterization trick
        batch_size = tf.shape(inputs)[0] # https://github.com/tensorflow/tensorflow/issues/373
        eps = tf.random_normal((batch_size, self.n_z), dtype=tf.float32)
        z = encoder_sigma*eps + encoder_mu
        if self.batch_norm:
            z_1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(z, self.z_1), is_training=self.training, updates_collections=None))
        else:
            z_1 = tf.nn.relu(tf.matmul(z, self.z_1) + self.z_1_b)

        # Input: latent variable (z) and previous hidden state
        decoder_input = tf.concat((z_1, h), 1)
        if self.batch_norm:
            decoder_h = tf.nn.relu(tf.contrib.layers.batch_norm(tf.matmul(decoder_input, self.decoder_h), is_training=self.training, updates_collections=None))
            decoder_sigma = tf.nn.softplus(tf.contrib.layers.batch_norm(tf.matmul(decoder_h, self.decoder_sigma), is_training=self.training, updates_collections=None))
        else:
            decoder_h = tf.nn.relu(tf.matmul(decoder_input, self.decoder_h) + self.decoder_h_b)
            decoder_sigma = tf.nn.softplus(tf.matmul(decoder_h, self.decoder_sigma) + self.decoder_sigma_b)
        decoder_mu = tf.matmul(decoder_h, self.decoder_mu) + self.decoder_mu_b

        # Pass to cell (e.g. LSTM). Note that the LSTM has both "h" and "c" that are combined
        # into the same next state vector. We'll combine them together to pass in and split them
        # back out after the LSTM returns the next state.
        rnn_cell_input = tf.concat((x_1, z_1), 1)
        _, (c_next, h_next) = self.cell(rnn_cell_input, [c, h]) # Note: (h,c) in Keras (c,h) in contrib

        # VRNN state
        next_state = (
            h_next,
            c_next,
            encoder_mu,
            encoder_sigma,
            decoder_mu,
            decoder_sigma,
            prior_mu,
            prior_sigma,
            x_1,
            z_1,
        )

        #return output, next_state
        return next_state, next_state