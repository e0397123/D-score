import tensorflow as tf


def bilstm_layer(input_data, num_layers, rnn_size, lengths, keep_prob=1.0):
    """Multi-layer BiLSTM
    Args:
        input_data: float32 Tensor of shape [seq_length, batch_size, dim].
        num_layers: int64 scalar, number of layers.
        rnn_size: int64 scalar, hidden size for undirectional LSTM.
        lengths: int64 Tensro of shape [batch_size]
        keep_prob: float32 scalar, keep probability of dropout between BiLSTM layers

    Return:
        hidden_state: float32 Tensor of shape [batch_size, dim * 2]

    """
    input_data = tf.transpose(input_data, [1, 0, 2])

    output = input_data

    hidden_state = None
    for layer in range(num_layers):
        with tf.variable_scope('bilstm_{}'.format(layer), reuse=tf.AUTO_REUSE):
            cell_fw = tf.contrib.rnn.LSTMCell(
                rnn_size, initializer=tf.truncated_normal_initializer(stddev=0.02))
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=keep_prob)

            cell_bw = tf.contrib.rnn.LSTMCell(
                rnn_size, initializer=tf.truncated_normal_initializer(stddev=0.02))
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=keep_prob)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                              cell_bw,
                                                              output,
                                                              sequence_length=lengths,
                                                              dtype=tf.float32)

            # Concat the forward and backward outputs
            output = tf.concat(outputs, 2)

            hidden_state = tf.concat([states[0].h, states[1].h], axis=1)

    output = tf.transpose(output, [1, 0, 2])

    return hidden_state, output


def downsample_embedding(inputs, dim=300):
    with tf.variable_scope('downsample_layer', reuse=tf.AUTO_REUSE):
        embed = tf.layers.dense(inputs, units=dim, kernel_initializer=tf.keras.initializers.glorot_normal())
        return embed


class HadeModel(object):

    def __init__(self, x_random_forward, x_random_mask_forward, x_random_length_forward, x_random_backward,
                 x_random_mask_backward, x_random_length_backward, y_random, x_swap_forward, x_swap_mask_forward,
                 x_swap_length_forward, x_swap_backward, x_swap_mask_backward, x_swap_length_backward, y_swap,
                 x_nli_forward, x_nli_mask_forward, x_nli_length_forward, x_nli_backward, x_nli_mask_backward,
                 x_nli_length_backward, y_nli, embedding_dim, num_nli_labels, hidden_size, l2_reg_lambda,
                 num_layers, dropout_rate, is_training):

        # random detection head
        self.x_random_forward = x_random_forward
        self.x_random_mask_forward = x_random_mask_forward
        self.x_random_length_forward = x_random_length_forward
        self.x_random_backward = x_random_backward
        self.x_random_mask_backward = x_random_mask_backward
        self.x_random_length_backward = x_random_length_backward
        self.y_random = y_random

        # coherence detection head
        self.x_swap_forward = x_swap_forward
        self.x_swap_mask_forward = x_swap_mask_forward
        self.x_swap_length_forward = x_swap_length_forward
        self.x_swap_backward = x_swap_backward
        self.x_swap_mask_backward = x_swap_mask_backward
        self.x_swap_length_backward = x_swap_length_backward
        self.y_swap = y_swap

        # # generic detection head
        # self.x_generic_forward = x_generic_forward
        # self.x_generic_mask_forward = x_generic_mask_forward
        # self.x_generic_length_forward = x_generic_length_forward
        # self.x_generic_backward = x_generic_backward
        # self.x_generic_mask_backward = x_generic_mask_backward
        # self.x_generic_length_backward = x_generic_length_backward
        # self.y_generic = y_generic

        # forward NLI detection head
        self.x_nli_forward = x_nli_forward
        self.x_nli_mask_forward = x_nli_mask_forward
        self.x_nli_length_forward = x_nli_length_forward

        # backward NLI detection head
        self.x_nli_backward = x_nli_backward
        self.x_nli_mask_backward = x_nli_mask_backward
        self.x_nli_length_backward = x_nli_length_backward
        self.y_nli = y_nli

        # other parameters
        self.embedding_dim = embedding_dim
        self.num_nli_labels = num_nli_labels
        self.hidden_size = hidden_size
        self.l2_reg_lambda = l2_reg_lambda
        self.num_layers = num_layers
        self.keep_rate = 1 - dropout_rate
        self.is_training = is_training

    def create_model(self):
        # embedding: [length, batch, dim]
        # random detection head
        emb_random_forward = downsample_embedding(self.x_random_forward)
        emb_random_backward = downsample_embedding(self.x_random_backward)
        if self.is_training:
            emb_random_forward = tf.nn.dropout(emb_random_forward, self.keep_rate)
            emb_random_backward = tf.nn.dropout(emb_random_backward, self.keep_rate)

        emb_random_forward = emb_random_forward * tf.expand_dims(self.x_random_mask_forward, -1)
        emb_random_backward = emb_random_backward * tf.expand_dims(self.x_random_mask_backward, -1)

        # swap detection head
        emb_swap_forward = downsample_embedding(self.x_swap_forward)
        emb_swap_backward = downsample_embedding(self.x_swap_backward)
        if self.is_training:
            emb_swap_forward = tf.nn.dropout(emb_swap_forward, self.keep_rate)
            emb_swap_backward = tf.nn.dropout(emb_swap_backward, self.keep_rate)

        emb_swap_forward = emb_swap_forward * tf.expand_dims(self.x_swap_mask_forward, -1)
        emb_swap_backward = emb_swap_backward * tf.expand_dims(self.x_swap_mask_backward, -1)

        # # generic detection head
        # emb_generic_forward = downsample_embedding(self.x_generic_forward)
        # emb_generic_backward = downsample_embedding(self.x_generic_backward)
        # if self.is_training:
        #     emb_generic_forward = tf.nn.dropout(emb_generic_forward, self.keep_rate)
        #     emb_generic_backward = tf.nn.dropout(emb_generic_backward, self.keep_rate)
        #
        # emb_generic_forward = emb_generic_forward * tf.expand_dims(self.x_generic_mask_forward, -1)
        # emb_generic_backward = emb_generic_backward * tf.expand_dims(self.x_generic_mask_backward, -1)

        # nli detection head
        emb_nli_forward = downsample_embedding(self.x_nli_forward)
        emb_nli_backward = downsample_embedding(self.x_nli_backward)
        if self.is_training:
            emb_nli_forward = tf.nn.dropout(emb_nli_forward, self.keep_rate)
            emb_nli_backward = tf.nn.dropout(emb_nli_backward, self.keep_rate)

        emb_nli_forward = emb_nli_forward * tf.expand_dims(self.x_nli_mask_forward, -1)
        emb_nli_backward = emb_nli_backward * tf.expand_dims(self.x_nli_mask_backward, -1)

        # encode the sentence pair
        with tf.variable_scope("lstm_encoder", reuse=tf.AUTO_REUSE):
            # [2*batch_size, 2*hidden state]
            # random detection head
            x1_random_enc, _ = bilstm_layer(emb_random_forward, self.num_layers, self.hidden_size,
                                            self.x_random_length_forward)
            x2_random_enc, _ = bilstm_layer(emb_random_backward, self.num_layers, self.hidden_size,
                                            self.x_random_length_backward)
            # swap detection head
            x1_swap_enc, _ = bilstm_layer(emb_swap_forward, self.num_layers, self.hidden_size,
                                          self.x_swap_length_forward)
            x2_swap_enc, _ = bilstm_layer(emb_swap_backward, self.num_layers, self.hidden_size,
                                          self.x_swap_length_backward)

            # # generic detection head
            # x1_generic_enc, _ = bilstm_layer(emb_generic_forward, self.num_layers, self.hidden_size,
            #                                  self.x_generic_length_forward)
            # x2_generic_enc, _ = bilstm_layer(emb_generic_backward, self.num_layers, self.hidden_size,
            #                                  self.x_generic_length_backward)

            # nli detection head
            x1_nli_enc, _ = bilstm_layer(emb_nli_forward, self.num_layers, self.hidden_size,
                                         self.x_nli_length_forward)
            x2_nli_enc, _ = bilstm_layer(emb_nli_backward, self.num_layers, self.hidden_size,
                                         self.x_nli_length_backward)

        with tf.variable_scope("matching_layer", reuse=tf.AUTO_REUSE):
            # random detection head
            m_random = tf.get_variable("M_random", shape=[2 * self.hidden_size, 2 * self.hidden_size],
                                       initializer=tf.truncated_normal_initializer())
            qtm_random = tf.tensordot(x1_random_enc, m_random, 1)
            # quadratic random feature
            quadratic_random = tf.reduce_sum(qtm_random * x2_random_enc, axis=1, keep_dims=True)
            # [2*batch_size, 4*hidden_size+1]
            concat_random = tf.concat([x1_random_enc, x2_random_enc, quadratic_random], axis=1)

            # swap detection head
            m_swap = tf.get_variable("M_swap", shape=[2 * self.hidden_size, 2 * self.hidden_size],
                                     initializer=tf.truncated_normal_initializer())
            qtm_swap = tf.tensordot(x1_swap_enc, m_swap, 1)
            # quadratic swap feature
            quadratic_swap = tf.reduce_sum(qtm_swap * x2_swap_enc, axis=1, keep_dims=True)
            # [2*batch_size, 4*hidden_size+1]
            concat_swap = tf.concat([x1_swap_enc, x2_swap_enc, quadratic_swap], axis=1)

            # # generic detection head
            # m_generic = tf.get_variable("M_generic", shape=[2 * self.hidden_size, 2 * self.hidden_size],
            #                             initializer=tf.truncated_normal_initializer())
            # qtm_generic = tf.tensordot(x1_generic_enc, m_generic, 1)
            # # quadratic generic feature
            # quadratic_generic = tf.reduce_sum(qtm_generic * x2_generic_enc, axis=1, keep_dims=True)
            # # [2*batch_size, 4*hidden_size+1]
            # concat_generic = tf.concat([x1_generic_enc, x2_generic_enc, quadratic_generic], axis=1)

            # nli detection head
            m_nli = tf.get_variable("M_nli", shape=[2 * self.hidden_size, 2 * self.hidden_size],
                                        initializer=tf.truncated_normal_initializer())
            qtm_nli = tf.tensordot(x1_nli_enc, m_nli, 1)
            # quadratic generic feature
            quadratic_nli = tf.reduce_sum(qtm_nli * x2_nli_enc, axis=1, keep_dims=True)
            # [2*batch_size, 4*hidden_size+1]
            concat_nli = tf.concat([x1_nli_enc, x2_nli_enc, quadratic_nli], axis=1)

        # random detection classifier
        with tf.variable_scope("random_classifier", reuse=tf.AUTO_REUSE):
            if self.is_training:
                random_logits = tf.nn.dropout(concat_random, self.keep_rate)
            else:
                random_logits = concat_random

            random_logits_1 = tf.layers.dense(random_logits, self.hidden_size, activation=tf.nn.tanh,
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            if self.is_training:
                random_logits_1 = tf.nn.dropout(random_logits_1, self.keep_rate)

            random_logits_2 = tf.layers.dense(random_logits_1, 2, activation=tf.nn.tanh,
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

        # swap detection classifier
        with tf.variable_scope("swap_classifier", reuse=tf.AUTO_REUSE):
            if self.is_training:
                swap_logits = tf.nn.dropout(concat_swap, self.keep_rate)
            else:
                swap_logits = concat_swap

            swap_logits_1 = tf.layers.dense(swap_logits, self.hidden_size, activation=tf.nn.tanh,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            if self.is_training:
                swap_logits_1 = tf.nn.dropout(swap_logits_1, self.keep_rate)

            swap_logits_2 = tf.layers.dense(swap_logits_1, 2, activation=tf.nn.tanh,
                                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

        # # generic detection classifier
        # with tf.variable_scope("generic_classifier", reuse=tf.AUTO_REUSE):
        #     if self.is_training:
        #         generic_logits = tf.nn.dropout(concat_generic, self.keep_rate)
        #     else:
        #         generic_logits = concat_generic
        #
        #     generic_logits_1 = tf.layers.dense(generic_logits, self.hidden_size, activation=tf.nn.tanh,
        #                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        #                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
        #     if self.is_training:
        #         generic_logits_1 = tf.nn.dropout(generic_logits_1, self.keep_rate)
        #
        #     generic_logits_2 = tf.layers.dense(generic_logits_1, 2, activation=tf.nn.tanh,
        #                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
        #                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

        # forward nli detection classifier
        with tf.variable_scope("nli_classifier", reuse=tf.AUTO_REUSE):
            if self.is_training:
                nli_logits = tf.nn.dropout(concat_nli, self.keep_rate)
            else:
                nli_logits = concat_nli

            nli_logits_1 = tf.layers.dense(nli_logits, self.hidden_size, activation=tf.nn.tanh,
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))
            if self.is_training:
                nli_logits_1 = tf.nn.dropout(nli_logits_1, self.keep_rate)

            nli_logits_2 = tf.layers.dense(nli_logits_1, self.num_nli_labels, activation=tf.nn.tanh,
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg_lambda))

        with tf.variable_scope("losses", reuse=tf.AUTO_REUSE):

            random_one_hot = tf.one_hot(self.y_random, depth=2, dtype=tf.float32)

            random_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=random_one_hot,
                                                                                 logits=random_logits_2))
            random_prob = tf.nn.softmax(random_logits_2)

            swap_one_hot = tf.one_hot(self.y_swap, depth=2, dtype=tf.float32)

            swap_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=swap_one_hot,
                                                                               logits=swap_logits_2))
            swap_prob = tf.nn.softmax(swap_logits_2)

            # generic_one_hot = tf.one_hot(self.y_generic, depth=2, dtype=tf.float32)
            #
            # generic_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=generic_one_hot,
            #                                                                       logits=random_logits_2))
            # generic_prob = tf.nn.softmax(generic_logits_2)

            # NLI related loss
            nli_one_hot = tf.one_hot(self.y_nli, depth=self.num_nli_labels, dtype=tf.float32)
            nli_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=nli_one_hot,
                                                                              logits=nli_logits_2))
            nli_prob = tf.nn.softmax(nli_logits_2)

            total_cost = random_cost + swap_cost + nli_cost

        return random_prob, swap_prob, nli_prob, total_cost

