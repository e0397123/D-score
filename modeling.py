"""
@Author:Zhang Chen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SimpleClassifier(nn.Module):
    """ a simple perceptron-based classifier """

    def __init__(self, num_features):
        """
        Args:
            num_features (int): the size of the input feature vector
        """
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features,
                             out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor
                x_in.shape should be (batch, num_features)
            apply_sigmoid (bool): a flag for the sigmoid activation
                should be false if used with the cross-entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch,).
        """
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = F.sigmoid(y_out)
        return y_out


class DscoreModel(nn.Module):
    """ DscoreModel Model """

    def __init__(self,
                 model,
                 encoder_dim,
                 hidden_dim,
                 dropout_rate,
                 num_lstm_layer,
                 device,
                 vocab_size,
                 num_random_classes=2,
                 num_swap_classes=2,
                 is_training=True,
                 bidirectional=True):
        """
        Args:
            model (transformer model): a pretrained transformer model
            lm_head: transformer lm head
            encoder_dim: output dimension of transformer encoder
            hidden_dim: dimension of hidden state
            dropout_rate: dropout rate
            num_lstm_layer: number of lstm layers
            device: cpu or gpu
            num_random_classes: number of random classes, default:2
            num_swap_classes: number of swap classes, default:2
            is_training: is_training or not
            bidirectional: whether use bidirectional lstm
        """
        super(DscoreModel, self).__init__()
        # forward(input_ids)
        self.device = device
        self.embeddings = model.embeddings
        self.encoder = model.encoder
        self.config = model.config
        self.is_training = is_training
        self.num_random_classes = num_random_classes
        self.num_swap_classes = num_swap_classes
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        self.downsample_layer = nn.Linear(in_features=encoder_dim,
                                          out_features=hidden_dim)
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim,
                            hidden_dim,
                            num_layers=num_lstm_layer,
                            bidirectional=bidirectional,
                            batch_first=True)
        self.num_directions = 2 if bidirectional else 1
        self.fc1_random = nn.Linear(hidden_dim * self.num_directions * 2 + 1, hidden_dim * self.num_directions)
        self.fc2_random = nn.Linear(hidden_dim * self.num_directions, hidden_dim)
        self.fc3_random = nn.Linear(hidden_dim, num_random_classes)
        self.fc1_swap = nn.Linear(hidden_dim * self.num_directions * 2 + 1, hidden_dim * self.num_directions)
        self.fc2_swap = nn.Linear(hidden_dim * self.num_directions, hidden_dim)
        self.fc3_swap = nn.Linear(hidden_dim, num_swap_classes)
        self.num_lstm_layer = num_lstm_layer
        self.lstm_units = hidden_dim
        self.m_random = Variable(torch.fmod(torch.empty([hidden_dim * self.num_directions,
                                                         hidden_dim * self.num_directions]), 2)).to(device)
        nn.init.normal_(self.m_random, std=0.02)

        self.m_swap = Variable(torch.fmod(torch.empty([hidden_dim * self.num_directions,
                                                       hidden_dim * self.num_directions]), 2)).to(device)
        nn.init.normal_(self.m_swap, std=0.02)

        self.gelu = nn.GELU()
        self.layernorm = nn.LayerNorm((hidden_dim, ), elementwise_affine=True)

        self.fc_lm = nn.Linear(hidden_dim, vocab_size)

    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.num_lstm_layer * self.num_directions, batch_size,
                                     self.lstm_units)).to(self.device),
                Variable(torch.zeros(self.num_lstm_layer * self.num_directions, batch_size,
                                     self.lstm_units)).to(self.device))
        return h, c

    def get_extended_attention_mask(self, attention_mask, input_shape, is_decoder=False):
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=self.device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_encoder_output(self, x, x_mask, is_decoder=False):
        embedding_outputs = self.embeddings(input_ids=x)
        head_mask = [None] * self.config.num_hidden_layers
        extended_attention_mask = self.get_extended_attention_mask(x_mask, x.shape, is_decoder=is_decoder)
        encoder_outputs = self.encoder(embedding_outputs, attention_mask=extended_attention_mask, head_mask=head_mask)
        return encoder_outputs[0]

    def forward(self, feature, batch_size):

        h_0, c_0 = self.init_hidden(batch_size)

        # --------------------------------------------------------------------------------------------------------------
        # random forward lstm output
        x_random_length_forward = feature["random_forward_text_len"]
        x_random_forward = feature["random_forward_input_ids"]
        x_random_mask_forward = feature["random_forward_input_mask"]
        # random_forward_bert_inputs = {'input_ids': x_random_forward, 'attention_mask': x_random_mask_forward}
        # [batch_size, max_pre, encoder_dim]
        random_forward_bert_output = self.get_encoder_output(x_random_forward,
                                                             x_random_mask_forward,
                                                             is_decoder=False)
        # [batch_size, max_pre, hidden_dim]
        random_forward_bert_output = self.downsample_layer(random_forward_bert_output)
        if self.is_training:
            random_forward_bert_output = self.dropout_layer(random_forward_bert_output)
        random_forward_packed_embedded = pack_padded_sequence(random_forward_bert_output,
                                                              x_random_length_forward,
                                                              batch_first=True,
                                                              enforce_sorted=False)
        random_forward_lstm_output, (_, _) = self.lstm(random_forward_packed_embedded, (h_0, c_0))
        random_forward_output_unpacked, random_forward_output_lengths = pad_packed_sequence(random_forward_lstm_output,
                                                                                            batch_first=True)
        # [batch_size, hidden_dim * 2]
        random_forward_out = random_forward_output_unpacked[:, -1, :]

        # --------------------------------------------------------------------------------------------------------------
        # random backward lstm output
        x_random_length_backward = feature["random_backward_text_len"]
        x_random_backward = feature["random_backward_input_ids"]
        x_random_mask_backward = feature["random_backward_input_mask"]
        # random_backward_bert_inputs = {'input_ids': x_random_backward, 'attention_mask': x_random_mask_backward}
        # [batch_size, max_post, encoder_dim]
        random_backward_bert_output = self.get_encoder_output(x_random_backward,
                                                             x_random_mask_backward,
                                                             is_decoder=False)
        # [batch_size, max_post, hidden_dim]
        random_backward_bert_output = self.downsample_layer(random_backward_bert_output)
        if self.is_training:
            random_backward_bert_output = self.dropout_layer(random_backward_bert_output)
        random_backward_packed_embedded = pack_padded_sequence(random_backward_bert_output,
                                                               x_random_length_backward,
                                                               batch_first=True,
                                                               enforce_sorted=False)
        random_backward_lstm_output, (_, _) = self.lstm(random_backward_packed_embedded,
                                                        (h_0, c_0))
        random_backward_output_unpacked, \
            random_backward_output_lengths = pad_packed_sequence(random_backward_lstm_output,
                                                                 batch_first=True)
        # [batch_size, hidden_dim * 2]
        random_backward_out = random_backward_output_unpacked[:, -1, :]

        # --------------------------------------------------------------------------------------------------------------
        # matching layer
        # [batch_size, hidden_dim*2]
        qtm_random = torch.tensordot(random_forward_out, self.m_random, dims=1)
        # quadratic random feature
        # [batch_size, hidden_dim*2]
        quadratic_random = torch.sum(qtm_random * random_backward_out, dim=1, keepdim=True)
        # [batch_size, 4*hidden_size+1]
        concat_random = torch.cat([random_forward_out, random_backward_out, quadratic_random], axis=1)

        # --------------------------------------------------------------------------------------------------------------
        # prediction layer
        rel_random = self.relu(concat_random)
        # [batch_size, 2*hidden_size]
        dense1_random = self.fc1_random(rel_random)
        if self.is_training:
            drop1_random = self.dropout_layer(dense1_random)
        else:
            drop1_random = dense1_random
        # [batch_size, hidden_size]
        dense2_random = self.fc2_random(drop1_random)
        if self.is_training:
            drop2_random = self.dropout_layer(dense2_random)
        else:
            drop2_random = dense2_random
        # [batch_size, num_classes]
        random_preds = self.fc3_random(drop2_random)

        random_prob = F.softmax(random_preds, dim=-1)

        # --------------------------------------------------------------------------------------------------------------
        # swap forward lstm output
        x_swap_length_forward = feature["swap_forward_text_len"]
        x_swap_forward = feature["swap_forward_input_ids"]
        x_swap_mask_forward = feature["swap_forward_input_mask"]
        # swap_forward_bert_inputs = {'input_ids': x_swap_forward, 'attention_mask': x_swap_mask_forward}
        # [batch_size, max_pre, encoder_dim]
        swap_forward_bert_output = self.get_encoder_output(x_swap_forward,
                                                           x_swap_mask_forward,
                                                           is_decoder=False)
        # [batch_size, max_pre, hidden_dim]
        swap_forward_bert_output = self.downsample_layer(swap_forward_bert_output)
        if self.is_training:
            swap_forward_bert_output = self.dropout_layer(swap_forward_bert_output)
        swap_forward_packed_embedded = pack_padded_sequence(swap_forward_bert_output,
                                                            x_swap_length_forward,
                                                            batch_first=True,
                                                            enforce_sorted=False)
        swap_forward_lstm_output, (_, _) = self.lstm(swap_forward_packed_embedded, (h_0, c_0))
        swap_forward_output_unpacked, swap_forward_output_lengths = pad_packed_sequence(swap_forward_lstm_output,
                                                                                        batch_first=True)
        # [batch_size, hidden_dim * 2]
        swap_forward_out = swap_forward_output_unpacked[:, -1, :]

        # --------------------------------------------------------------------------------------------------------------
        # swap backward lstm output
        x_swap_length_backward = feature["swap_backward_text_len"]
        x_swap_backward = feature["swap_backward_input_ids"]
        x_swap_mask_backward = feature["swap_backward_segment_ids"]
        # [batch_size, max_post, encoder_dim]
        swap_backward_bert_output = self.get_encoder_output(x_swap_backward,
                                                           x_swap_mask_backward,
                                                           is_decoder=False)
        # [batch_size, max_post, hidden_dim]
        swap_backward_bert_output = self.downsample_layer(swap_backward_bert_output)
        if self.is_training:
            swap_backward_bert_output = self.dropout_layer(swap_backward_bert_output)
        swap_backward_packed_embedded = pack_padded_sequence(swap_backward_bert_output,
                                                             x_swap_length_backward,
                                                             batch_first=True,
                                                             enforce_sorted=False)
        swap_backward_lstm_output, (_, _) = self.lstm(swap_backward_packed_embedded, (h_0, c_0))
        swap_backward_output_unpacked, swap_backward_output_lengths = pad_packed_sequence(swap_backward_lstm_output,
                                                                                          batch_first=True)
        # [batch_size, hidden_dim * 2]
        swap_backward_out = swap_backward_output_unpacked[:, -1, :]

        # --------------------------------------------------------------------------------------------------------------
        # matching layer
        # [batch_size, hidden_dim*2]
        qtm_swap = torch.tensordot(swap_forward_out, self.m_swap, dims=1)
        # quadratic swap feature
        # [batch_size, hidden_dim*2]
        quadratic_swap = torch.sum(qtm_swap * swap_backward_out, dim=1, keepdim=True)
        # [batch_size, 4*hidden_size+1]
        concat_swap = torch.cat([swap_forward_out, swap_backward_out, quadratic_swap], axis=1)

        # --------------------------------------------------------------------------------------------------------------
        # prediction layer
        rel_swap = self.relu(concat_swap)
        # [batch_size, 2*hidden_size]
        dense1_swap = self.fc1_swap(rel_swap)
        if self.is_training:
            drop1_swap = self.dropout_layer(dense1_swap)
        else:
            drop1_swap = dense1_swap
        # [batch_size, hidden_size]
        dense2_swap = self.fc2_swap(drop1_swap)
        if self.is_training:
            drop2_swap = self.dropout_layer(dense2_swap)
        else:
            drop2_swap = dense2_swap
        # [batch_size, num_classes]
        swap_preds = self.fc3_swap(drop2_swap)

        swap_prob = F.softmax(swap_preds, dim=-1)

        # LM head
        x_lm_sequence = feature["response_input_ids"]
        x_lm_sequence_mask = feature["response_input_mask"]
        # [batch_size, max_pre + max_post, encoder_dim]
        lm_sequence_bert_output = self.get_encoder_output(x_lm_sequence,
                                                          x_lm_sequence_mask,
                                                          is_decoder=True)
        # [batch_size, max_pre + max_post, hidden_dim]
        lm_sequence_bert_output = self.downsample_layer(lm_sequence_bert_output)
        if self.is_training:
            lm_sequence_bert_output = self.dropout_layer(lm_sequence_bert_output)

        response_logits = self.gelu(lm_sequence_bert_output)
        response_logits = self.layernorm(response_logits)
        # [batch_size, max_pre + max_post, vocab_size]
        response_outputs = self.fc_lm(response_logits)

        return random_preds, random_prob, swap_preds, swap_prob, response_outputs
