import tensorflow as tf
import copy

from bert import modeling
from modeling import HadeModel


__all__ = ['InputExample', 'InputFeatures', 'create_bilstm_classification_model']


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, utterance_id, correct_pre_context, correct_post_context,
                 correct_current_response, swapped_context, swapped_current_response,
                 dull_response, random_utt, swap_pre, rand_current):

        self.guid = guid
        self.utterance_id = utterance_id
        self.correct_pre_context = correct_pre_context
        self.correct_post_context = correct_post_context
        self.correct_current_response = correct_current_response
        self.swapped_context = swapped_context
        self.swapped_current_response = swapped_current_response
        self.dull_response = dull_response
        self.random_utt = random_utt
        self.swap_pre = swap_pre
        self.rand_current = rand_current


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 response_input_ids,
                 response_input_mask,
                 response_segment_ids,
                 response_labels,
                 response_text_len,
                 random_forward_input_ids,
                 random_forward_input_mask,
                 random_forward_segment_ids,
                 random_forward_text_len,
                 random_backward_input_ids,
                 random_backward_input_mask,
                 random_backward_segment_ids,
                 random_backward_text_len,
                 random_labels,
                 swap_forward_input_ids,
                 swap_forward_input_mask,
                 swap_forward_segment_ids,
                 swap_forward_text_len,
                 swap_backward_input_ids,
                 swap_backward_input_mask,
                 swap_backward_segment_ids,
                 swap_backward_text_len,
                 swap_labels,
                 # generic_forward_input_ids,
                 # generic_forward_input_mask,
                 # generic_forward_segment_ids,
                 # generic_forward_text_len,
                 # generic_backward_input_ids,
                 # generic_backward_input_mask,
                 # generic_backward_segment_ids,
                 # generic_backward_text_len,
                 # generic_labels,
                 nli_forward_input_ids,
                 nli_forward_input_mask,
                 nli_forward_segment_ids,
                 nli_forward_text_len,
                 nli_backward_input_ids,
                 nli_backward_input_mask,
                 nli_backward_segment_ids,
                 nli_backward_text_len,
                 nli_labels,
                 current_utterance_id):

        self.response_input_ids = response_input_ids
        self.response_input_mask = response_input_mask
        self.response_segment_ids = response_segment_ids
        self.response_labels = response_labels
        self.response_text_len = response_text_len
        self.random_forward_input_ids = random_forward_input_ids
        self.random_forward_input_mask = random_forward_input_mask
        self.random_forward_segment_ids = random_forward_segment_ids
        self.random_forward_text_len = random_forward_text_len
        self.random_backward_input_ids = random_backward_input_ids
        self.random_backward_input_mask = random_backward_input_mask
        self.random_backward_segment_ids = random_backward_segment_ids
        self.random_backward_text_len = random_backward_text_len
        self.random_labels = random_labels
        self.swap_forward_input_ids = swap_forward_input_ids
        self.swap_forward_input_mask = swap_forward_input_mask
        self.swap_forward_segment_ids = swap_forward_segment_ids
        self.swap_forward_text_len = swap_forward_text_len
        self.swap_backward_input_ids = swap_backward_input_ids
        self.swap_backward_input_mask = swap_backward_input_mask
        self.swap_backward_segment_ids = swap_backward_segment_ids
        self.swap_backward_text_len = swap_backward_text_len
        self.swap_labels = swap_labels
        # self.generic_forward_input_ids = generic_forward_input_ids
        # self.generic_forward_input_mask = generic_forward_input_mask
        # self.generic_forward_segment_ids = generic_forward_segment_ids
        # self.generic_forward_text_len = generic_forward_text_len
        # self.generic_backward_input_ids = generic_backward_input_ids
        # self.generic_backward_input_mask = generic_backward_input_mask
        # self.generic_backward_segment_ids = generic_backward_segment_ids
        # self.generic_backward_text_len = generic_backward_text_len
        # self.generic_labels = generic_labels
        self.nli_forward_input_ids = nli_forward_input_ids
        self.nli_forward_input_mask = nli_forward_input_mask
        self.nli_forward_segment_ids = nli_forward_segment_ids
        self.nli_forward_text_len = nli_forward_text_len
        self.nli_backward_input_ids = nli_backward_input_ids
        self.nli_backward_input_mask = nli_backward_input_mask
        self.nli_backward_segment_ids = nli_backward_segment_ids
        self.nli_backward_text_len = nli_backward_text_len
        self.nli_labels = nli_labels
        self.current_utterance_id = current_utterance_id


def create_bilstm_classification_model(bert_config, is_training, response_input_ids, response_input_mask,
                                       response_segment_ids, response_text_len, response_labels,
                                       random_forward_input_ids, random_forward_input_mask,
                                       random_forward_segment_ids, random_forward_text_len, random_backward_input_ids,
                                       random_backward_input_mask, random_backward_segment_ids, random_backward_text_len,
                                       random_labels, swap_forward_input_ids, swap_forward_input_mask,
                                       swap_forward_segment_ids, swap_forward_text_len, swap_backward_input_ids,
                                       swap_backward_input_mask, swap_backward_segment_ids, swap_backward_text_len,
                                       swap_labels, nli_forward_input_ids, nli_forward_input_mask,
                                       nli_forward_segment_ids, nli_forward_text_len, nli_backward_input_ids,
                                       nli_backward_input_mask, nli_backward_segment_ids, nli_backward_text_len,
                                       nli_labels, num_nli_labels, use_one_hot_embeddings,
                                       l2_reg_lambda=0.1, dropout_rate=1.0, lstm_size=None, num_layers=1):

    config = copy.deepcopy(bert_config)

    if not is_training:
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0

    with tf.variable_scope("bert", reuse=tf.AUTO_REUSE):

        with tf.variable_scope("embeddings", reuse=tf.AUTO_REUSE):
            (response_embedding_output, response_embedding_table) = modeling.embedding_lookup(
                input_ids=response_input_ids,
                vocab_size=config.vocab_size,
                embedding_size=config.hidden_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings)

            response_embedding_output = modeling.embedding_postprocessor(
                input_tensor=response_embedding_output,
                use_token_type=not config.roberta,
                token_type_ids=response_segment_ids,
                token_type_vocab_size=config.type_vocab_size,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings,
                dropout_prob=config.hidden_dropout_prob,
                roberta=config.roberta)

            # random detection
            # Perform embedding lookup on the word ids.
            (random_foward_embedding_output, random_forward_embedding_table) = modeling.embedding_lookup(
                input_ids=random_forward_input_ids,
                vocab_size=config.vocab_size,
                embedding_size=config.hidden_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings)

            # Perform embedding lookup on the word ids.
            (random_backward_embedding_output, random_backward_embedding_table) = modeling.embedding_lookup(
                input_ids=random_backward_input_ids,
                vocab_size=config.vocab_size,
                embedding_size=config.hidden_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings)

            # Add positional embeddings and token type embeddings, then layer
            # normalize and perform dropout.
            random_foward_embedding_output = modeling.embedding_postprocessor(
                input_tensor=random_foward_embedding_output,
                use_token_type=not config.roberta,
                token_type_ids=random_forward_segment_ids,
                token_type_vocab_size=config.type_vocab_size,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings,
                dropout_prob=config.hidden_dropout_prob,
                roberta=config.roberta)

            random_backward_embedding_output = modeling.embedding_postprocessor(
                input_tensor=random_backward_embedding_output,
                use_token_type=not config.roberta,
                token_type_ids=random_backward_segment_ids,
                token_type_vocab_size=config.type_vocab_size,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings,
                dropout_prob=config.hidden_dropout_prob,
                roberta=config.roberta)

            # swap detection
            (swap_foward_embedding_output, swap_forward_embedding_table) = modeling.embedding_lookup(
                input_ids=swap_forward_input_ids,
                vocab_size=config.vocab_size,
                embedding_size=config.hidden_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings)

            (swap_backward_embedding_output, swap_backward_embedding_table) = modeling.embedding_lookup(
                input_ids=swap_backward_input_ids,
                vocab_size=config.vocab_size,
                embedding_size=config.hidden_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings)
            swap_foward_embedding_output = modeling.embedding_postprocessor(
                input_tensor=swap_foward_embedding_output,
                use_token_type=not config.roberta,
                token_type_ids=swap_forward_segment_ids,
                token_type_vocab_size=config.type_vocab_size,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings,
                dropout_prob=config.hidden_dropout_prob,
                roberta=config.roberta)
            swap_backward_embedding_output = modeling.embedding_postprocessor(
                input_tensor=swap_backward_embedding_output,
                use_token_type=not config.roberta,
                token_type_ids=swap_backward_segment_ids,
                token_type_vocab_size=config.type_vocab_size,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings,
                dropout_prob=config.hidden_dropout_prob,
                roberta=config.roberta)

            # # generic detection
            # (generic_foward_embedding_output, generic_forward_embedding_table) = modeling.embedding_lookup(
            #     input_ids=generic_forward_input_ids,
            #     vocab_size=config.vocab_size,
            #     embedding_size=config.hidden_size,
            #     initializer_range=config.initializer_range,
            #     word_embedding_name="word_embeddings",
            #     use_one_hot_embeddings=use_one_hot_embeddings)
            # (generic_backward_embedding_output, generic_backward_embedding_table) = modeling.embedding_lookup(
            #     input_ids=generic_backward_input_ids,
            #     vocab_size=config.vocab_size,
            #     embedding_size=config.hidden_size,
            #     initializer_range=config.initializer_range,
            #     word_embedding_name="word_embeddings",
            #     use_one_hot_embeddings=use_one_hot_embeddings)
            # generic_foward_embedding_output = modeling.embedding_postprocessor(
            #     input_tensor=generic_foward_embedding_output,
            #     use_token_type=not config.roberta,
            #     token_type_ids=generic_forward_segment_ids,
            #     token_type_vocab_size=config.type_vocab_size,
            #     token_type_embedding_name="token_type_embeddings",
            #     use_position_embeddings=True,
            #     position_embedding_name="position_embeddings",
            #     initializer_range=config.initializer_range,
            #     max_position_embeddings=config.max_position_embeddings,
            #     dropout_prob=config.hidden_dropout_prob,
            #     roberta=config.roberta)
            # generic_backward_embedding_output = modeling.embedding_postprocessor(
            #     input_tensor=generic_backward_embedding_output,
            #     use_token_type=not config.roberta,
            #     token_type_ids=generic_backward_segment_ids,
            #     token_type_vocab_size=config.type_vocab_size,
            #     token_type_embedding_name="token_type_embeddings",
            #     use_position_embeddings=True,
            #     position_embedding_name="position_embeddings",
            #     initializer_range=config.initializer_range,
            #     max_position_embeddings=config.max_position_embeddings,
            #     dropout_prob=config.hidden_dropout_prob,
            #     roberta=config.roberta)

            # nli detection
            (nli_foward_embedding_output, nli_forward_embedding_table) = modeling.embedding_lookup(
                input_ids=nli_forward_input_ids,
                vocab_size=config.vocab_size,
                embedding_size=config.hidden_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings)
            (nli_backward_embedding_output, nli_backward_embedding_table) = modeling.embedding_lookup(
                input_ids=nli_backward_input_ids,
                vocab_size=config.vocab_size,
                embedding_size=config.hidden_size,
                initializer_range=config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=use_one_hot_embeddings)
            nli_foward_embedding_output = modeling.embedding_postprocessor(
                input_tensor=nli_foward_embedding_output,
                use_token_type=not config.roberta,
                token_type_ids=nli_forward_segment_ids,
                token_type_vocab_size=config.type_vocab_size,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings,
                dropout_prob=config.hidden_dropout_prob,
                roberta=config.roberta)
            nli_backward_embedding_output = modeling.embedding_postprocessor(
                input_tensor=nli_backward_embedding_output,
                use_token_type=not config.roberta,
                token_type_ids=nli_backward_segment_ids,
                token_type_vocab_size=config.type_vocab_size,
                token_type_embedding_name="token_type_embeddings",
                use_position_embeddings=True,
                position_embedding_name="position_embeddings",
                initializer_range=config.initializer_range,
                max_position_embeddings=config.max_position_embeddings,
                dropout_prob=config.hidden_dropout_prob,
                roberta=config.roberta)

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            response_attention_mask = modeling.create_attention_mask_from_input_mask(response_input_ids,
                                                                                     response_input_mask)
            # [batch_size, from_seq_length, to_seq_length]
            # mask future tokens
            diag_vals = tf.ones_like(response_attention_mask[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(response_attention_mask)[0], 1, 1])
            response_attention_mask = tf.math.multiply(response_attention_mask, future_masks)
            # Run the stacked transformer.
            # `sequence_output` shape = [batch_size, seq_length, hidden_size].
            response_all_encoder_layers = modeling.transformer_model(
                input_tensor=response_embedding_output,
                attention_mask=response_attention_mask,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                intermediate_act_fn=modeling.get_activation(config.hidden_act),
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                do_return_all_layers=True)


            # random detection
            # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
            # mask of shape [batch_size, seq_length, seq_length] which is used
            # for the attention scores.
            random_forward_attention_mask = modeling.create_attention_mask_from_input_mask(random_forward_input_ids,
                                                                                           random_forward_input_mask)
            random_backward_attention_mask = modeling.create_attention_mask_from_input_mask(random_backward_input_ids,
                                                                                            random_backward_input_mask)
            # Run the stacked transformer.
            # `sequence_output` shape = [batch_size, seq_length, hidden_size].
            random_forward_all_encoder_layers = modeling.transformer_model(
                input_tensor=random_foward_embedding_output,
                attention_mask=random_forward_attention_mask,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                intermediate_act_fn=modeling.get_activation(config.hidden_act),
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                do_return_all_layers=True)
            random_backward_all_encoder_layers = modeling.transformer_model(
                input_tensor=random_backward_embedding_output,
                attention_mask=random_backward_attention_mask,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                intermediate_act_fn=modeling.get_activation(config.hidden_act),
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                do_return_all_layers=True)

            # swap detection
            swap_forward_attention_mask = modeling.create_attention_mask_from_input_mask(swap_forward_input_ids,
                                                                                         swap_forward_input_mask)
            swap_backward_attention_mask = modeling.create_attention_mask_from_input_mask(swap_backward_input_ids,
                                                                                          swap_backward_input_mask)
            swap_forward_all_encoder_layers = modeling.transformer_model(
                input_tensor=swap_foward_embedding_output,
                attention_mask=swap_forward_attention_mask,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                intermediate_act_fn=modeling.get_activation(config.hidden_act),
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                do_return_all_layers=True)
            swap_backward_all_encoder_layers = modeling.transformer_model(
                input_tensor=swap_backward_embedding_output,
                attention_mask=swap_backward_attention_mask,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                intermediate_act_fn=modeling.get_activation(config.hidden_act),
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                do_return_all_layers=True)

            # # generic detection
            # generic_forward_attention_mask = modeling.create_attention_mask_from_input_mask(generic_forward_input_ids,
            #                                                                                 generic_forward_input_mask)
            # generic_backward_attention_mask = modeling.create_attention_mask_from_input_mask(generic_backward_input_ids,
            #                                                                                  generic_backward_input_mask)
            # generic_forward_all_encoder_layers = modeling.transformer_model(
            #     input_tensor=generic_foward_embedding_output,
            #     attention_mask=generic_forward_attention_mask,
            #     hidden_size=config.hidden_size,
            #     num_hidden_layers=config.num_hidden_layers,
            #     num_attention_heads=config.num_attention_heads,
            #     intermediate_size=config.intermediate_size,
            #     intermediate_act_fn=modeling.get_activation(config.hidden_act),
            #     hidden_dropout_prob=config.hidden_dropout_prob,
            #     attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            #     initializer_range=config.initializer_range,
            #     do_return_all_layers=True)
            # generic_backward_all_encoder_layers = modeling.transformer_model(
            #     input_tensor=generic_backward_embedding_output,
            #     attention_mask=generic_backward_attention_mask,
            #     hidden_size=config.hidden_size,
            #     num_hidden_layers=config.num_hidden_layers,
            #     num_attention_heads=config.num_attention_heads,
            #     intermediate_size=config.intermediate_size,
            #     intermediate_act_fn=modeling.get_activation(config.hidden_act),
            #     hidden_dropout_prob=config.hidden_dropout_prob,
            #     attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            #     initializer_range=config.initializer_range,
            #     do_return_all_layers=True)

            # nli detection
            nli_forward_attention_mask = modeling.create_attention_mask_from_input_mask(nli_forward_input_ids,
                                                                                        nli_forward_input_mask)
            nli_backward_attention_mask = modeling.create_attention_mask_from_input_mask(nli_backward_input_ids,
                                                                                         nli_backward_input_mask)
            nli_forward_all_encoder_layers = modeling.transformer_model(
                input_tensor=nli_foward_embedding_output,
                attention_mask=nli_forward_attention_mask,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                intermediate_act_fn=modeling.get_activation(config.hidden_act),
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                do_return_all_layers=True)
            nli_backward_all_encoder_layers = modeling.transformer_model(
                input_tensor=nli_backward_embedding_output,
                attention_mask=nli_backward_attention_mask,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                intermediate_act_fn=modeling.get_activation(config.hidden_act),
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                do_return_all_layers=True)

        random_forward_embedding = random_forward_all_encoder_layers[-2]
        random_backward_embedding = random_backward_all_encoder_layers[-2]
        swap_forward_embedding = swap_forward_all_encoder_layers[-2]
        swap_backward_embedding = swap_backward_all_encoder_layers[-2]
        # generic_forward_embedding = generic_forward_all_encoder_layers[-2]
        # generic_backward_embedding = generic_backward_all_encoder_layers[-2]
        nli_forward_embedding = nli_forward_all_encoder_layers[-2]
        nli_backward_embedding = nli_backward_all_encoder_layers[-2]
        response_embedding = response_all_encoder_layers[-2]

    response_embedding_shape = modeling.get_shape_list(response_embedding, expected_rank=3)
    with tf.variable_scope("lm_head", reuse=tf.AUTO_REUSE):

        response_logits = tf.layers.dense(response_embedding, config.hidden_size, activation=None)
        response_logits = modeling.gelu(response_logits)
        response_logits = modeling.layer_norm(response_logits)
        response_outputs = tf.layers.dense(response_logits, config.vocab_size,
                                            activation=None, use_bias=True, bias_initializer=tf.zeros_initializer())

        response_one_hot = tf.one_hot(response_labels, depth=config.vocab_size, dtype=tf.float32)

        lm_cost = tf.nn.softmax_cross_entropy_with_logits(labels=response_one_hot, logits=response_outputs)

        sequence_mask = tf.sequence_mask(response_text_len, maxlen=response_embedding_shape[1], dtype=tf.float32)

        masked_lm_cost = tf.math.multiply(lm_cost, sequence_mask)

        final_lm_loss = tf.reduce_mean(tf.math.divide(tf.reduce_sum(masked_lm_cost, axis=1), tf.cast(response_text_len, dtype=tf.float32)))

        perplexity = tf.exp(tf.math.divide(tf.reduce_sum(masked_lm_cost, axis=1), tf.cast(response_text_len, dtype=tf.float32)))

    random_forward_embedding_shape = modeling.get_shape_list(random_forward_embedding, expected_rank=3)
    random_backward_embedding_shape = modeling.get_shape_list(random_backward_embedding, expected_rank=3)
    assert random_forward_embedding_shape[2] == random_backward_embedding_shape[2]
    random_forward_embedding = tf.transpose(random_forward_embedding, [1, 0, 2])
    random_backward_embedding = tf.transpose(random_backward_embedding, [1, 0, 2])
    random_forward_input_mask = tf.cast(tf.transpose(random_forward_input_mask, [1, 0]), tf.float32)
    random_backward_input_mask = tf.cast(tf.transpose(random_backward_input_mask, [1, 0]), tf.float32)

    swap_forward_embedding_shape = modeling.get_shape_list(swap_forward_embedding, expected_rank=3)
    swap_backward_embedding_shape = modeling.get_shape_list(swap_backward_embedding, expected_rank=3)
    assert swap_forward_embedding_shape[2] == swap_backward_embedding_shape[2]
    swap_forward_embedding = tf.transpose(swap_forward_embedding, [1, 0, 2])
    swap_backward_embedding = tf.transpose(swap_backward_embedding, [1, 0, 2])
    swap_forward_input_mask = tf.cast(tf.transpose(swap_forward_input_mask, [1, 0]), tf.float32)
    swap_backward_input_mask = tf.cast(tf.transpose(swap_backward_input_mask, [1, 0]), tf.float32)

    # generic_forward_embedding_shape = modeling.get_shape_list(generic_forward_embedding, expected_rank=3)
    # generic_backward_embedding_shape = modeling.get_shape_list(generic_backward_embedding, expected_rank=3)
    # assert generic_forward_embedding_shape[2] == generic_backward_embedding_shape[2]
    # generic_forward_embedding = tf.transpose(generic_forward_embedding, [1, 0, 2])
    # generic_backward_embedding = tf.transpose(generic_backward_embedding, [1, 0, 2])
    # generic_forward_input_mask = tf.cast(tf.transpose(generic_forward_input_mask, [1, 0]), tf.float32)
    # generic_backward_input_mask = tf.cast(tf.transpose(generic_backward_input_mask, [1, 0]), tf.float32)

    nli_forward_embedding_shape = modeling.get_shape_list(nli_forward_embedding, expected_rank=3)
    nli_backward_embedding_shape = modeling.get_shape_list(nli_backward_embedding, expected_rank=3)
    assert nli_forward_embedding_shape[2] == nli_backward_embedding_shape[2]
    nli_forward_embedding = tf.transpose(nli_forward_embedding, [1, 0, 2])
    nli_backward_embedding = tf.transpose(nli_backward_embedding, [1, 0, 2])
    nli_forward_input_mask = tf.cast(tf.transpose(nli_forward_input_mask, [1, 0]), tf.float32)
    nli_backward_input_mask = tf.cast(tf.transpose(nli_backward_input_mask, [1, 0]), tf.float32)

    model = HadeModel(x_random_forward=random_forward_embedding, x_random_mask_forward=random_forward_input_mask,
                      x_random_length_forward=random_forward_text_len, x_random_backward=random_backward_embedding,
                      x_random_mask_backward=random_backward_input_mask,
                      x_random_length_backward=random_backward_text_len,
                      y_random=random_labels, x_swap_forward=swap_forward_embedding,
                      x_swap_mask_forward=swap_forward_input_mask,
                      x_swap_length_forward=swap_forward_text_len, x_swap_backward=swap_backward_embedding,
                      x_swap_mask_backward=swap_backward_input_mask, x_swap_length_backward=swap_backward_text_len,
                      y_swap=swap_labels,
                      # x_generic_forward=generic_forward_embedding,
                      # x_generic_mask_forward=generic_forward_input_mask,
                      # x_generic_length_forward=generic_forward_text_len,
                      # x_generic_backward=generic_backward_embedding,
                      # x_generic_mask_backward=generic_backward_input_mask,
                      # x_generic_length_backward=generic_backward_text_len, y_generic=generic_labels,
                      x_nli_forward=nli_forward_embedding,x_nli_mask_forward=nli_forward_input_mask,
                      x_nli_length_forward=nli_forward_text_len, x_nli_backward=nli_backward_embedding,
                      x_nli_mask_backward=nli_backward_input_mask,
                      x_nli_length_backward=nli_backward_text_len, y_nli=nli_labels,
                      embedding_dim=random_forward_embedding_shape[2], num_nli_labels=num_nli_labels,
                      hidden_size=lstm_size, l2_reg_lambda=l2_reg_lambda, num_layers=num_layers,
                      dropout_rate=dropout_rate, is_training=is_training)

    random_prob, swap_prob, nli_prob, total_cost = model.create_model()

    return random_prob, swap_prob, nli_prob, total_cost, final_lm_loss, perplexity
