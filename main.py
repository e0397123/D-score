#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_ALBERT.

Modified script originally created by @Author: Macan

@Modified by:Zhang Chen
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf
import codecs
import pickle
import pandas as pd
import random
import sys

from best_checkpoint_copier import BestCheckpointCopier
from tqdm import tqdm

from bert import modeling
from bert import optimization
from transformers import RobertaTokenizer as tokenization

from unified_framework import create_bilstm_classification_model, InputFeatures, InputExample
from tf_metrics import precision, recall, f1

random.seed(2000)

__all__ = ['NspProcessor', 'convert_single_example',
           'filed_based_convert_examples_to_features', 'file_based_input_fn_builder',
           'model_fn_builder', 'main']

flags = tf.flags

FLAGS = flags.FLAGS

bert_path = '../pretrained_module/roberta/roberta-base-pretrained-tf'

root_path = '/home/chen/hade_main'

flags.DEFINE_string('data_dir', default=os.path.join(root_path, 'datasets/persona-chat/'),
                    help="train, dev and test data dir")

flags.DEFINE_string('roberta_config_file', default=os.path.join(bert_path, 'roberta-base-config.json'),
                    help="roberta config file path")

flags.DEFINE_string('output_dir', default=os.path.join(root_path, 'nsp_models'),
                    help='directory of trained model')

flags.DEFINE_string('init_checkpoint', default=os.path.join(bert_path, 'roberta_base.ckpt'),
                    help='Initial checkpoint (usually from a pre-trained model).')

flags.DEFINE_string('corpus_name', default='persona', help='corpus name')

flags.DEFINE_string('task', default='nsp', help='which model to train')

flags.DEFINE_string('eval_type', default='eval', help='specify type of data to evaluate')

flags.DEFINE_integer('dupe_factor', default=1, help='number of times to duplicate input')

flags.DEFINE_integer('max_pre_len', default=256,
                     help='The maximum total input sequence length after Sentencepiece tokenization.')

flags.DEFINE_integer('max_post_len', default=256,
                     help='The maximum total input sequence length after Sentencepiece tokenization.')

flags.DEFINE_integer('max_seq_len', default=256,
                     help='The maximum total response sequence length after Sentencepiece tokenization.')

flags.DEFINE_integer('window_size', default=5, help='number of utterances in a context window')

flags.DEFINE_integer('batch_size', default=32, help='Total batch size for training, eval and predict.')

flags.DEFINE_integer('num_train_epochs', default=10, help='Total number of training epochs to perform.')

flags.DEFINE_integer('lstm_size', default=300, help='size of lstm units.')

flags.DEFINE_integer('num_layers', default=1, help='number of rnn layers, default is 1.')

flags.DEFINE_integer('keep_checkpoint_max', default=3, help='number of checkpoints to keep, default is 3.')

flags.DEFINE_integer('save_checkpoints_steps', default=2000, help='save_checkpoints_steps')

flags.DEFINE_integer('save_summary_steps', default=2000, help='save_summary_steps.')

flags.DEFINE_integer('valid_steps', default=100, help='save_summary_steps.')

flags.DEFINE_float('learning_rate', default=1e-5, help='The initial learning rate for Adam.')

flags.DEFINE_float('dropout_rate', default=0.5, help='Dropout rate')

flags.DEFINE_float('l2_reg_lambda', default=0.2, help='l2_reg_lambda')

flags.DEFINE_float('warmup_proportion', default=0.025,
                   help='Proportion of training to perform linear learning rate warmup for '
                        'E.g., 0.1 = 10% of training.')

flags.DEFINE_bool('do_train', default=False, help='Whether to run training.')

flags.DEFINE_bool('do_eval', default=False, help='Whether to run eval on the dev set.')

flags.DEFINE_bool('do_predict', default=False, help='Whether to run the predict in inference mode on the test set.')

flags.DEFINE_bool('filter_adam_var', default=False,
                  help='after training do filter Adam params from model and save no Adam params model in file.')

flags.DEFINE_bool('do_lower_case', default=True, help='Whether to lower case the input text.')

flags.DEFINE_bool('clean', default=False, help="whether to clean output folder")

logger = tf.get_logger()

logger.propagate = False

tokenizer = tokenization.from_pretrained('roberta-base')


class NspProcessor(object):

    def __init__(self, output_dir):
        self.labels = []
        self.output_dir = output_dir
        with codecs.open('dull_responses.txt', mode='r', encoding='utf-8') as rf:
            lines = rf.readlines()
        self.dull_responses = [l.strip() for l in lines]

    def process_dialogue(self, data_dir, corpus_name, split):
        logger.info("*************** reading {} dialogue data*****************************".format(split))
        dialogue_data = pd.read_csv(os.path.join(data_dir, corpus_name + '_main.csv'))
        meta_data = pd.read_csv(os.path.join(data_dir, corpus_name + '_metadata.csv'))
        if corpus_name == 'dstc6' and (split == 'train' or split == 'valid'):
            cond_a = meta_data['type'] == split
            cond_b = meta_data['num_turn'] > 3
            meta_data = meta_data[cond_a & cond_b]
        dialogue_ids = list(meta_data[meta_data['type'] == split]['dialogue_id'])
        dialogues = []
        dialogue_uids = []
        for i in tqdm(dialogue_ids):
            single_dialogue = [str(item) for item in list(dialogue_data[dialogue_data['UID'].str.startswith(i)]['SEG'])]
            single_dialogue_uid = list(dialogue_data[dialogue_data['UID'].str.startswith(i)]['UID'])
            single_dialogue.insert(0, 'start of dialogue')
            single_dialogue_uid.insert(0, corpus_name + '-' + 'start-placeholder')
            single_dialogue.append('end of dialogue')
            single_dialogue_uid.append(corpus_name + '-' + 'end-placeholder')
            dialogues.append(single_dialogue)
            dialogue_uids.append(single_dialogue_uid)
        return dialogues, dialogue_uids

    def get_labels(self):
        self.labels.append('original')
        self.labels.append('swap')
        self.labels.append('dull')
        self.labels.append('random')
        return self.labels

    def get_eval_utterance_labels(self, data_dir, corpus_name):
        dialogue_data = pd.read_csv(os.path.join(data_dir, corpus_name + '_main.csv'))
        utterance_ids = list(dialogue_data['UID'])
        return utterance_ids

    def get_train_example(self, train_dialogues, train_dialogue_uids, context_window_size, split='train'):
        if os.path.exists(os.path.join(FLAGS.output_dir, '{}_lines.pkl'.format(split))):
            return self._create_example(pickle.load(open(os.path.join(FLAGS.output_dir,
                                                                      '{}_lines.pkl'.format(split)), 'rb')), split)
        else:
            lines = self._read_data(train_dialogues, train_dialogue_uids, context_window_size, split)
            pickle.dump(lines, open(os.path.join(FLAGS.output_dir, 'train_lines.pkl'), 'wb'))
            return self._create_example(lines, split)

    def get_valid_example(self, valid_dialogues, valid_dialogue_uids, context_window_size, split='valid'):
        if os.path.exists(os.path.join(FLAGS.output_dir, '{}_lines.pkl'.format(split))):
            return self._create_example(pickle.load(open(os.path.join(FLAGS.output_dir,
                                                                      'valid_lines.pkl'), 'rb')), split)
        else:
            lines = self._read_data(valid_dialogues, valid_dialogue_uids, context_window_size, split)
            pickle.dump(lines, open(os.path.join(FLAGS.output_dir, '{}_lines.pkl'.format(split)), 'wb'))
            return self._create_example(lines, split)

    def get_eval_example(self, eval_dialogues, eval_dialogue_uids, context_window_size, split):
        if os.path.exists(os.path.join(FLAGS.output_dir, '{}_lines.pkl'.format(split))):
            return self._create_example(pickle.load(open(os.path.join(FLAGS.output_dir,
                                                                      '{}_lines.pkl'.format(split)), 'rb')), split)
        else:
            lines = self._read_data(eval_dialogues, eval_dialogue_uids, context_window_size, split)
            pickle.dump(lines, open(os.path.join(FLAGS.output_dir, '{}_lines.pkl'.format(split)), 'wb'))
            return self._create_example(lines, split)

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            utterance_id = line[0]
            correct_pre_context = line[1]
            correct_post_context = line[2]
            correct_current_response = line[3]
            swapped_context = line[4]
            swapped_current_response = line[5]
            dull_response = line[6]
            random_utt = line[7]
            swap_pre = line[8]
            rand_current = line[9]
            # if i == 0:
            #     logger.info('label: ', label)
            examples.append(InputExample(guid=guid, utterance_id=utterance_id,
                                         correct_pre_context=correct_pre_context,
                                         correct_post_context=correct_post_context,
                                         correct_current_response=correct_current_response,
                                         swapped_context=swapped_context,
                                         swapped_current_response=swapped_current_response,
                                         dull_response=dull_response,
                                         random_utt=random_utt,
                                         swap_pre=swap_pre,
                                         rand_current=rand_current))
        return examples

    def _read_data(self, dialogues, dialogue_uids, context_window_size, split):
        if context_window_size <= 2:
            raise ValueError("window size must be larger than 2")
        logger.info("*************** form {} dialogue triplet *****************************".format(split))
        lines = []
        if split == 'train' or split == 'valid':
            for t in range(FLAGS.dupe_factor):
                for idx, d in enumerate(tqdm(dialogues)):
                    if len(d) < context_window_size:
                        raise ValueError("length of dialogue {0} is less than "
                                         "window size of {1}".format(idx, context_window_size))
                    elif len(d) == context_window_size:
                        for i in range(1, len(d) - 1, 1):
                            current_utt = d[i]
                            pre_context = d[:i]
                            post_context = d[i + 1:]
                            utterance_idx = dialogue_uids[idx][i]
                            correct_pre_context = ' </s> '.join(pre_context)
                            correct_post_context = ' </s> '.join(post_context)
                            correct_current_response = current_utt

                            # swap current utt with context utt
                            rand_num = random.random()
                            if rand_num < 0.5:
                                random_idx = random.choice(list(range(0, i, 1)))
                                pre_context_swapped = [item if item != d[random_idx] else current_utt for item in
                                                       pre_context]
                                current_utt_swapped = d[random_idx]
                                swapped_context = ' </s> '.join(pre_context_swapped)
                                swapped_current_response = current_utt_swapped
                                swap_pre = True
                            else:
                                # swap current utt with one in the post-context
                                random_idx = random.choice(list(range(i + 1, len(d), 1)))
                                post_context_swapped = [item if item != d[random_idx] else current_utt for item in
                                                        post_context]
                                current_utt_swapped = d[random_idx]

                                swapped_context = ' </s> '.join(post_context_swapped)
                                swapped_current_response = current_utt_swapped
                                swap_pre = False

                            # generic response
                            dull_response = random.choice(self.dull_responses)

                            rand_num = random.random()
                            if rand_num < 0.5:
                                # random choice from other dialogues
                                random_dialogue_idx = random.choice(list(range(0, len(dialogues), 1)))
                                while random_dialogue_idx == idx:
                                    random_dialogue_idx = random.choice(list(range(0, len(dialogues), 1)))
                                random_utt = random.choice(dialogues[random_dialogue_idx])
                                while random_utt == current_utt:
                                    random_utt = random.choice(dialogues[random_dialogue_idx])
                                rand_current = False
                            else:
                                # random choice from current dialogue
                                random_utt = random.choice(d)
                                while random_utt == current_utt:
                                    random_utt = random.choice(d)
                                splitted_random_utt = random_utt.split(' ')
                                random.shuffle(splitted_random_utt)
                                random_utt = ' '.join(splitted_random_utt)
                                rand_current = True

                            lines.append([utterance_idx, correct_pre_context, correct_post_context,
                                          correct_current_response, swapped_context, swapped_current_response,
                                          dull_response, random_utt, swap_pre, rand_current])
                    else:
                        for j in range(0, len(d) - context_window_size + 1, 1):
                            window_sized_d = d[j:j + context_window_size]
                            for k in range(1, len(window_sized_d) - 1, 1):
                                current_utt = window_sized_d[k]
                                pre_context = window_sized_d[:k]
                                post_context = window_sized_d[k + 1:]
                                utterance_idx = dialogue_uids[idx][j + k]
                                correct_pre_context = ' </s> '.join(pre_context)
                                correct_post_context = ' </s> '.join(post_context)
                                correct_current_response = current_utt

                                # swap current utt with one in the pre-context
                                rand_num = random.random()
                                if rand_num < 0.5:
                                    random_idx = random.choice(list(range(0, k, 1)))
                                    pre_context_swapped = [item if item != window_sized_d[random_idx]
                                                           else current_utt for item in pre_context]
                                    current_utt_swapped = window_sized_d[random_idx]

                                    swapped_context = ' </s> '.join(pre_context_swapped)
                                    swapped_current_response = current_utt_swapped
                                    swap_pre = True
                                else:
                                    # swap current utt with one in the post-context
                                    random_idx = random.choice(list(range(k + 1, len(window_sized_d), 1)))
                                    post_context_swapped = [item if item != window_sized_d[random_idx]
                                                            else current_utt for item in post_context]
                                    current_utt_swapped = window_sized_d[random_idx]

                                    swapped_context = ' </s> '.join(post_context_swapped)
                                    swapped_current_response = current_utt_swapped
                                    swap_pre = False

                                # generic response
                                dull_response = random.choice(self.dull_responses)

                                # random choice from other dialogues
                                rand_num = random.random()
                                if rand_num < 0.5:
                                    random_dialogue_idx = random.choice(list(range(0, len(dialogues), 1)))
                                    while random_dialogue_idx == idx:
                                        random_dialogue_idx = random.choice(list(range(0, len(dialogues), 1)))
                                    random_utt = random.choice(dialogues[random_dialogue_idx])
                                    while random_utt == current_utt:
                                        random_utt = random.choice(dialogues[random_dialogue_idx])
                                    rand_current = False
                                else:
                                    # random choice from current dialogue
                                    random_utt = random.choice(d)
                                    while random_utt == current_utt:
                                        random_utt = random.choice(d)
                                    splitted_random_utt = random_utt.split(' ')
                                    random.shuffle(splitted_random_utt)
                                    random_utt = ' '.join(splitted_random_utt)
                                    rand_current = True

                                lines.append([utterance_idx, correct_pre_context, correct_post_context,
                                              correct_current_response, swapped_context, swapped_current_response,
                                              dull_response, random_utt, swap_pre, rand_current])
        else:
            for idx, d in enumerate(tqdm(dialogues)):
                if len(d) < context_window_size:
                    raise ValueError("length of dialogue {0} is less "
                                     "than window size of {1}".format(idx, context_window_size))
                elif len(d) == context_window_size:
                    for i in range(1, len(d) - 1, 1):
                        current_utt = d[i]
                        pre_context = d[:i]
                        post_context = d[i + 1:]

                        utterance_idx = dialogue_uids[idx][i]
                        correct_pre_context = ' </s> '.join(pre_context)
                        correct_post_context = ' </s> '.join(post_context)
                        correct_current_response = current_utt

                        lines.append([utterance_idx, correct_pre_context, correct_post_context,
                                      correct_current_response, correct_pre_context, correct_current_response,
                                      correct_current_response, correct_current_response, False, False])

                else:
                    for j in range(0, len(d) - context_window_size + 1, 1):
                        window_sized_d = d[j:j + context_window_size]
                        for k in range(1, len(window_sized_d) - 1, 1):
                            current_utt = window_sized_d[k]
                            pre_context = window_sized_d[:k]
                            post_context = window_sized_d[k + 1:]
                            utterance_idx = dialogue_uids[idx][j + k]
                            correct_pre_context = ' </s> '.join(pre_context)
                            correct_post_context = ' </s> '.join(post_context)
                            correct_current_response = current_utt

                            lines.append([utterance_idx, correct_pre_context, correct_post_context,
                                          correct_current_response, correct_pre_context, correct_current_response,
                                          correct_current_response, correct_current_response, False, False])
        return lines


def _truncate_seq_pair_back(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens)
        if total_length <= max_length:
            break
        else:
            tokens.pop(-2)


def _truncate_seq_pair_front(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens)
        if total_length <= max_length:
            break
        else:
            tokens.pop(1)


def convert_single_example(ex_index, example, utterance_label_map,
                           max_pre_len, max_post_len, max_seq_len, is_training=True):
    # random detection
    if is_training:
        rand_num = random.random()
    else:
        rand_num = 0

    if rand_num < 0.5:
        b_input_ids = tokenizer.encode(example.correct_current_response)
        random_labels = 1
    else:
        b_input_ids = tokenizer.encode(example.random_utt)
        random_labels = 0

    # pre context
    a_input_ids = tokenizer.encode(example.correct_pre_context)
    # current response

    random_forward_input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=a_input_ids[1:-1],
                                                                          token_ids_1=b_input_ids[1:-1])
    random_forward_segment_ids = tokenizer.create_token_type_ids_from_sequences(token_ids_0=a_input_ids[1:-1],
                                                                                token_ids_1=b_input_ids[1:-1])
    _truncate_seq_pair_front(random_forward_input_ids, max_pre_len)
    _truncate_seq_pair_front(random_forward_segment_ids, max_pre_len)

    # post context
    c_input_ids = tokenizer.encode(example.correct_post_context)
    random_backward_input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=b_input_ids[1:-1],
                                                                           token_ids_1=c_input_ids[1:-1])

    random_backward_segment_ids = tokenizer.create_token_type_ids_from_sequences(token_ids_0=b_input_ids[1:-1],
                                                                                 token_ids_1=c_input_ids[1:-1])
    _truncate_seq_pair_back(random_backward_input_ids, max_post_len)
    _truncate_seq_pair_back(random_backward_segment_ids, max_post_len)

    random_forward_text_len = len(random_forward_segment_ids)
    random_backward_text_len = len(random_backward_segment_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    random_forward_input_mask = [1] * len(random_forward_input_ids)
    random_backward_input_mask = [1] * len(random_backward_input_ids)

    # Zero-pad up to the sequence length.
    while len(random_forward_input_ids) < max_pre_len:
        # pad token id is 1
        random_forward_input_ids.append(1)
        random_forward_input_mask.append(0)
        random_forward_segment_ids.append(0)

    assert len(random_forward_input_ids) == max_pre_len
    assert len(random_forward_input_mask) == max_pre_len
    assert len(random_forward_segment_ids) == max_pre_len

    while len(random_backward_input_ids) < max_post_len:
        # pad token id is 1
        random_backward_input_ids.append(1)
        random_backward_input_mask.append(0)
        random_backward_segment_ids.append(0)

    assert len(random_backward_input_ids) == max_post_len
    assert len(random_backward_input_mask) == max_post_len
    assert len(random_backward_segment_ids) == max_post_len

    # LM head
    org_response_input_ids = tokenizer.encode(example.correct_current_response)
    _truncate_seq_pair_back(org_response_input_ids, max_seq_len)

    response_input_ids = org_response_input_ids[:-1]
    response_labels = org_response_input_ids[1:]
    response_input_mask = [1] * len(response_input_ids)
    response_segment_ids = [1] * len(response_input_ids)
    response_text_len = len(response_input_ids)

    # Zero-pad up to the sequence length.
    while len(response_input_ids) < max_seq_len:
        # pad token id is 1
        response_input_ids.append(1)
        response_input_mask.append(0)
        response_segment_ids.append(0)
        response_labels.append(1)

    assert len(response_input_ids) == max_seq_len
    assert len(response_input_mask) == max_seq_len
    assert len(response_segment_ids) == max_seq_len
    assert len(response_labels) == max_seq_len

    # swap detection
    swap_pre = example.swap_pre
    if is_training:
        rand_num = random.random()
    else:
        rand_num = 0
    if swap_pre:
        if rand_num < 0.5:
            a_input_ids = tokenizer.encode(example.correct_pre_context)
            b_input_ids = tokenizer.encode(example.correct_current_response)
            c_input_ids = tokenizer.encode(example.correct_post_context)
            swap_labels = 1
        else:
            a_input_ids = tokenizer.encode(example.swapped_context)
            b_input_ids = tokenizer.encode(example.swapped_current_response)
            c_input_ids = tokenizer.encode(example.correct_post_context)
            swap_labels = 0
    else:
        if rand_num < 0.5:
            a_input_ids = tokenizer.encode(example.correct_pre_context)
            b_input_ids = tokenizer.encode(example.correct_current_response)
            c_input_ids = tokenizer.encode(example.correct_post_context)
            swap_labels = 1
        else:
            a_input_ids = tokenizer.encode(example.correct_pre_context)
            b_input_ids = tokenizer.encode(example.swapped_current_response)
            c_input_ids = tokenizer.encode(example.swapped_context)
            swap_labels = 0

    swap_forward_input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=a_input_ids[1:-1],
                                                                        token_ids_1=b_input_ids[1:-1])
    swap_forward_segment_ids = tokenizer.create_token_type_ids_from_sequences(token_ids_0=a_input_ids[1:-1],
                                                                              token_ids_1=b_input_ids[1:-1])
    _truncate_seq_pair_front(swap_forward_input_ids, max_pre_len)
    _truncate_seq_pair_front(swap_forward_segment_ids, max_pre_len)

    swap_backward_input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=b_input_ids[1:-1],
                                                                         token_ids_1=c_input_ids[1:-1])

    swap_backward_segment_ids = tokenizer.create_token_type_ids_from_sequences(token_ids_0=b_input_ids[1:-1],
                                                                               token_ids_1=c_input_ids[1:-1])
    _truncate_seq_pair_back(swap_backward_input_ids, max_post_len)
    _truncate_seq_pair_back(swap_backward_segment_ids, max_post_len)

    swap_forward_text_len = len(swap_forward_segment_ids)
    swap_backward_text_len = len(swap_backward_segment_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    swap_forward_input_mask = [1] * len(swap_forward_input_ids)
    swap_backward_input_mask = [1] * len(swap_backward_input_ids)

    # Zero-pad up to the sequence length.
    while len(swap_forward_input_ids) < max_pre_len:
        # pad token id is 1
        swap_forward_input_ids.append(1)
        swap_forward_input_mask.append(0)
        swap_forward_segment_ids.append(0)

    assert len(swap_forward_input_ids) == max_pre_len
    assert len(swap_forward_input_mask) == max_pre_len
    assert len(swap_forward_segment_ids) == max_pre_len

    while len(swap_backward_input_ids) < max_post_len:
        # pad token id is 1
        swap_backward_input_ids.append(1)
        swap_backward_input_mask.append(0)
        swap_backward_segment_ids.append(0)

    assert len(swap_backward_input_ids) == max_post_len
    assert len(swap_backward_input_mask) == max_post_len
    assert len(swap_backward_segment_ids) == max_post_len

    # # generic detection
    # if is_training:
    #     rand_num = random.random()
    # else:
    #     rand_num = 0
    #
    # if rand_num < 0.3:
    #     b_input_ids = tokenizer.encode(example.correct_current_response)
    #     generic_labels = 1
    # elif rand_num < 0.6:
    #     altered_response = example.correct_current_response.split()
    #     if len(altered_response) > 2:
    #         random.shuffle(altered_response)
    #         altered_response_drop = ' '.join(random.sample(altered_response, k=(len(altered_response) - 2)))
    #         b_input_ids = tokenizer.encode(altered_response_drop)
    #         generic_labels = 0
    #     else:
    #         b_input_ids = tokenizer.encode(example.correct_current_response)
    #         generic_labels = 1
    # else:
    #     b_input_ids = tokenizer.encode(example.dull_response)
    #     generic_labels = 0
    #
    # # pre context
    # a_input_ids = tokenizer.encode(example.correct_pre_context)
    # # current response
    #
    # generic_forward_input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=a_input_ids[1:-1],
    #                                                                        token_ids_1=b_input_ids[1:-1])
    # generic_forward_segment_ids = tokenizer.create_token_type_ids_from_sequences(token_ids_0=a_input_ids[1:-1],
    #                                                                              token_ids_1=b_input_ids[1:-1])
    # _truncate_seq_pair_front(generic_forward_input_ids, max_pre_len)
    # _truncate_seq_pair_front(generic_forward_segment_ids, max_pre_len)
    # # post context
    # c_input_ids = tokenizer.encode(example.correct_post_context)
    # generic_backward_input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=b_input_ids[1:-1],
    #                                                                         token_ids_1=c_input_ids[1:-1])
    #
    # generic_backward_segment_ids = tokenizer.create_token_type_ids_from_sequences(token_ids_0=b_input_ids[1:-1],
    #                                                                               token_ids_1=c_input_ids[1:-1])
    # _truncate_seq_pair_back(generic_backward_input_ids, max_post_len)
    # _truncate_seq_pair_back(generic_backward_segment_ids, max_post_len)
    #
    # generic_forward_text_len = len(generic_forward_segment_ids)
    # generic_backward_text_len = len(generic_backward_segment_ids)
    #
    # # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # # tokens are attended to.
    # generic_forward_input_mask = [1] * len(generic_forward_input_ids)
    # generic_backward_input_mask = [1] * len(generic_backward_input_ids)
    #
    # # Zero-pad up to the sequence length.
    # while len(generic_forward_input_ids) < max_pre_len:
    #     # pad token id is 1
    #     generic_forward_input_ids.append(1)
    #     generic_forward_input_mask.append(0)
    #     generic_forward_segment_ids.append(0)
    #
    # assert len(generic_forward_input_ids) == max_pre_len
    # assert len(generic_forward_input_mask) == max_pre_len
    # assert len(generic_forward_segment_ids) == max_pre_len
    #
    # while len(generic_backward_input_ids) < max_post_len:
    #     # pad token id is 1
    #     generic_backward_input_ids.append(1)
    #     generic_backward_input_mask.append(0)
    #     generic_backward_segment_ids.append(0)
    #
    # assert len(generic_backward_input_ids) == max_post_len
    # assert len(generic_backward_input_mask) == max_post_len
    # assert len(generic_backward_segment_ids) == max_post_len

    # NLI detection
    if is_training:
        rand_num = random.random()
    else:
        rand_num = 0
    rand_current = example.rand_current
    if rand_num < 0.25:
        a_input_ids = tokenizer.encode(example.correct_pre_context)
        b_input_ids = tokenizer.encode(example.correct_current_response)
        c_input_ids = tokenizer.encode(example.correct_post_context)
        nli_labels = 2
    elif rand_num <= 0.50:
        a_input_ids = tokenizer.encode(example.correct_pre_context)
        b_input_ids = tokenizer.encode(example.dull_response)
        c_input_ids = tokenizer.encode(example.correct_post_context)
        nli_labels = 1
    elif rand_num <= 0.75:
        a_input_ids = tokenizer.encode(example.correct_pre_context)
        b_input_ids = tokenizer.encode(example.swapped_current_response)
        c_input_ids = tokenizer.encode(example.correct_post_context)
        nli_labels = 2
    else:
        a_input_ids = tokenizer.encode(example.correct_pre_context)
        b_input_ids = tokenizer.encode(example.random_utt)
        c_input_ids = tokenizer.encode(example.correct_post_context)
        if rand_current:
            nli_labels = 0
        else:
            nli_labels = 1

    nli_forward_input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=a_input_ids[1:-1],
                                                                       token_ids_1=b_input_ids[1:-1])
    nli_forward_segment_ids = tokenizer.create_token_type_ids_from_sequences(token_ids_0=a_input_ids[1:-1],
                                                                             token_ids_1=b_input_ids[1:-1])
    _truncate_seq_pair_front(nli_forward_input_ids, max_pre_len)
    _truncate_seq_pair_front(nli_forward_segment_ids, max_pre_len)

    nli_backward_input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_0=b_input_ids[1:-1],
                                                                        token_ids_1=c_input_ids[1:-1])

    nli_backward_segment_ids = tokenizer.create_token_type_ids_from_sequences(token_ids_0=b_input_ids[1:-1],
                                                                              token_ids_1=c_input_ids[1:-1])
    _truncate_seq_pair_back(nli_backward_input_ids, max_post_len)
    _truncate_seq_pair_back(nli_backward_segment_ids, max_post_len)

    nli_forward_text_len = len(nli_forward_segment_ids)
    nli_backward_text_len = len(nli_backward_segment_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    nli_forward_input_mask = [1] * len(nli_forward_input_ids)
    nli_backward_input_mask = [1] * len(nli_backward_input_ids)

    # Zero-pad up to the sequence length.
    while len(nli_forward_input_ids) < max_pre_len:
        # pad token id is 1
        nli_forward_input_ids.append(1)
        nli_forward_input_mask.append(0)
        nli_forward_segment_ids.append(0)

    assert len(nli_forward_input_ids) == max_pre_len
    assert len(nli_forward_input_mask) == max_pre_len
    assert len(nli_forward_segment_ids) == max_pre_len

    while len(nli_backward_input_ids) < max_post_len:
        # pad token id is 1
        nli_backward_input_ids.append(1)
        nli_backward_input_mask.append(0)
        nli_backward_segment_ids.append(0)

    assert len(nli_backward_input_ids) == max_post_len
    assert len(nli_backward_input_mask) == max_post_len
    assert len(nli_backward_segment_ids) == max_post_len

    current_utterance_id = utterance_label_map[example.utterance_id]

    if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("unique_id: %s" % example.guid)
        logger.info("current_utterance_id: %s" % current_utterance_id)
        logger.info("response sequence length: %s" % str(response_text_len))
        logger.info("response sequence input_ids: %s" % " ".join([str(x) for x in response_input_ids]))
        logger.info("response sequence input_mask: %s" % " ".join([str(x) for x in response_input_mask]))
        logger.info("response sequence input_type_ids: %s" % " ".join([str(x) for x in response_segment_ids]))
        logger.info("random_forward sequence length: %s" % str(random_forward_text_len))
        logger.info("random_forward sequence input_ids: %s" % " ".join([str(x) for x in random_forward_input_ids]))
        logger.info("random_forward sequence input_mask: %s" % " ".join([str(x) for x in random_forward_input_mask]))
        logger.info(
            "random_forward sequence input_type_ids: %s" % " ".join([str(x) for x in random_forward_segment_ids]))
        logger.info("random_backward sequence length: %s" % str(random_backward_text_len))
        logger.info("random_backward sequence input_ids: %s" % " ".join([str(x) for x in random_backward_input_ids]))
        logger.info("random_backward sequence input_mask: %s" % " ".join([str(x) for x in random_backward_input_mask]))
        logger.info(
            "random_backward sequence input_type_ids: %s" % " ".join([str(x) for x in random_backward_segment_ids]))
        logger.info("random label: %s" % str(random_labels))
        logger.info("swap_forward sequence length: %s" % str(swap_forward_text_len))
        logger.info("swap_forward sequence input_ids: %s" % " ".join([str(x) for x in swap_forward_input_ids]))
        logger.info("swap_forward sequence input_mask: %s" % " ".join([str(x) for x in swap_forward_input_mask]))
        logger.info("swap_forward sequence input_type_ids: %s" % " ".join([str(x) for x in swap_forward_segment_ids]))
        logger.info("swap_backward sequence length: %s" % str(swap_backward_text_len))
        logger.info("swap_backward sequence input_ids: %s" % " ".join([str(x) for x in swap_backward_input_ids]))
        logger.info("swap_backward sequence input_mask: %s" % " ".join([str(x) for x in swap_backward_input_mask]))
        logger.info("swap_backward sequence input_type_ids: %s" % " ".join([str(x) for x in swap_backward_segment_ids]))
        logger.info("swap label: %s" % str(swap_labels))
        # logger.info("generic_forward sequence length: %s" % str(generic_forward_text_len))
        # logger.info("generic_forward sequence input_ids: %s" % " ".join([str(x) for x in generic_forward_input_ids]))
        # logger.info("generic_forward sequence input_mask: %s" % " ".join([str(x) for x in generic_forward_input_mask]))
        # logger.info(
        #     "generic_forward sequence input_type_ids: %s" % " ".join([str(x) for x in generic_forward_segment_ids]))
        # logger.info("generic_backward sequence length: %s" % str(generic_backward_text_len))
        # logger.info("generic_backward sequence input_ids: %s" % " ".join([str(x) for x in generic_backward_input_ids]))
        # logger.info(
        #     "generic_backward sequence input_mask: %s" % " ".join([str(x) for x in generic_backward_input_mask]))
        # logger.info(
        #     "generic_backward sequence input_type_ids: %s" % " ".join([str(x) for x in generic_backward_segment_ids]))
        # logger.info("generic label: %s" % str(generic_labels))
        logger.info("nli_forward sequence length: %s" % str(nli_forward_text_len))
        logger.info("nli_forward sequence input_ids: %s" % " ".join([str(x) for x in nli_forward_input_ids]))
        logger.info("nli_forward sequence input_mask: %s" % " ".join([str(x) for x in nli_forward_input_mask]))
        logger.info("nli_forward sequence input_type_ids: %s" % " ".join([str(x) for x in nli_forward_segment_ids]))
        logger.info("nli_backward sequence length: %s" % str(nli_backward_text_len))
        logger.info("nli_backward sequence input_ids: %s" % " ".join([str(x) for x in nli_backward_input_ids]))
        logger.info("nli_backward sequence input_mask: %s" % " ".join([str(x) for x in nli_backward_input_mask]))
        logger.info("nli_backward sequence input_type_ids: %s" % " ".join([str(x) for x in nli_backward_segment_ids]))
        logger.info("nli label: %s" % str(nli_labels))

    # 结构化为一个类
    feature = InputFeatures(
        response_input_ids=response_input_ids,
        response_input_mask=response_input_mask,
        response_segment_ids=response_segment_ids,
        response_labels=response_labels,
        response_text_len=response_text_len,
        random_forward_input_ids=random_forward_input_ids,
        random_forward_input_mask=random_forward_input_mask,
        random_forward_segment_ids=random_forward_segment_ids,
        random_forward_text_len=random_forward_text_len,
        random_backward_input_ids=random_backward_input_ids,
        random_backward_input_mask=random_backward_input_mask,
        random_backward_segment_ids=random_backward_segment_ids,
        random_backward_text_len=random_backward_text_len,
        random_labels=random_labels,
        swap_forward_input_ids=swap_forward_input_ids,
        swap_forward_input_mask=swap_forward_input_mask,
        swap_forward_segment_ids=swap_forward_segment_ids,
        swap_forward_text_len=swap_forward_text_len,
        swap_backward_input_ids=swap_backward_input_ids,
        swap_backward_input_mask=swap_backward_input_mask,
        swap_backward_segment_ids=swap_backward_segment_ids,
        swap_backward_text_len=swap_backward_text_len,
        swap_labels=swap_labels,
        # generic_forward_input_ids=generic_forward_input_ids,
        # generic_forward_input_mask=generic_forward_input_mask,
        # generic_forward_segment_ids=generic_forward_segment_ids,
        # generic_forward_text_len=generic_forward_text_len,
        # generic_backward_input_ids=generic_backward_input_ids,
        # generic_backward_input_mask=generic_backward_input_mask,
        # generic_backward_segment_ids=generic_backward_segment_ids,
        # generic_backward_text_len=generic_backward_text_len,
        # generic_labels=generic_labels,
        nli_forward_input_ids=nli_forward_input_ids,
        nli_forward_input_mask=nli_forward_input_mask,
        nli_forward_segment_ids=nli_forward_segment_ids,
        nli_forward_text_len=nli_forward_text_len,
        nli_backward_input_ids=nli_backward_input_ids,
        nli_backward_input_mask=nli_backward_input_mask,
        nli_backward_segment_ids=nli_backward_segment_ids,
        nli_backward_text_len=nli_backward_text_len,
        nli_labels=nli_labels,
        current_utterance_id=current_utterance_id
    )

    return feature


def filed_based_convert_examples_to_features(
        examples, utterance_label_map, max_pre_len, max_post_len, max_seq_len, output_file, is_training=True):
    writer = tf.python_io.TFRecordWriter(output_file)
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        # 对于每一个训练样本,
        feature = convert_single_example(ex_index, example, utterance_label_map,
                                         max_pre_len, max_post_len, max_seq_len, is_training=is_training)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["response_input_ids"] = create_int_feature(feature.response_input_ids)
        features["response_input_mask"] = create_int_feature(feature.response_input_mask)
        features["response_segment_ids"] = create_int_feature(feature.response_segment_ids)
        features["response_labels"] = create_int_feature(feature.response_labels)
        features["response_text_len"] = create_int_feature([feature.response_text_len])
        features["random_forward_input_ids"] = create_int_feature(feature.random_forward_input_ids)
        features["random_forward_input_mask"] = create_int_feature(feature.random_forward_input_mask)
        features["random_forward_segment_ids"] = create_int_feature(feature.random_forward_segment_ids)
        features["random_forward_text_len"] = create_int_feature([feature.random_forward_text_len])
        features["random_backward_input_ids"] = create_int_feature(feature.random_backward_input_ids)
        features["random_backward_input_mask"] = create_int_feature(feature.random_backward_input_mask)
        features["random_backward_segment_ids"] = create_int_feature(feature.random_backward_segment_ids)
        features["random_backward_text_len"] = create_int_feature([feature.random_backward_text_len])
        features["random_labels"] = create_int_feature([feature.random_labels])
        features["swap_forward_input_ids"] = create_int_feature(feature.swap_forward_input_ids)
        features["swap_forward_input_mask"] = create_int_feature(feature.swap_forward_input_mask)
        features["swap_forward_segment_ids"] = create_int_feature(feature.swap_forward_segment_ids)
        features["swap_forward_text_len"] = create_int_feature([feature.swap_forward_text_len])
        features["swap_backward_input_ids"] = create_int_feature(feature.swap_backward_input_ids)
        features["swap_backward_input_mask"] = create_int_feature(feature.swap_backward_input_mask)
        features["swap_backward_segment_ids"] = create_int_feature(feature.swap_backward_segment_ids)
        features["swap_backward_text_len"] = create_int_feature([feature.swap_backward_text_len])
        features["swap_labels"] = create_int_feature([feature.swap_labels])
        # features["generic_forward_input_ids"] = create_int_feature(feature.generic_forward_input_ids)
        # features["generic_forward_input_mask"] = create_int_feature(feature.generic_forward_input_mask)
        # features["generic_forward_segment_ids"] = create_int_feature(feature.generic_forward_segment_ids)
        # features["generic_forward_text_len"] = create_int_feature([feature.generic_forward_text_len])
        # features["generic_backward_input_ids"] = create_int_feature(feature.generic_backward_input_ids)
        # features["generic_backward_input_mask"] = create_int_feature(feature.generic_backward_input_mask)
        # features["generic_backward_segment_ids"] = create_int_feature(feature.generic_backward_segment_ids)
        # features["generic_backward_text_len"] = create_int_feature([feature.generic_backward_text_len])
        # features["generic_labels"] = create_int_feature([feature.generic_labels])
        features["nli_forward_input_ids"] = create_int_feature(feature.nli_forward_input_ids)
        features["nli_forward_input_mask"] = create_int_feature(feature.nli_forward_input_mask)
        features["nli_forward_segment_ids"] = create_int_feature(feature.nli_forward_segment_ids)
        features["nli_forward_text_len"] = create_int_feature([feature.nli_forward_text_len])
        features["nli_backward_input_ids"] = create_int_feature(feature.nli_backward_input_ids)
        features["nli_backward_input_mask"] = create_int_feature(feature.nli_backward_input_mask)
        features["nli_backward_segment_ids"] = create_int_feature(feature.nli_backward_segment_ids)
        features["nli_backward_text_len"] = create_int_feature([feature.nli_backward_text_len])
        features["nli_labels"] = create_int_feature([feature.nli_labels])
        features["current_utterance_id"] = create_int_feature([feature.current_utterance_id])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, a_seq_length, b_seq_length, c_seq_length, is_training, drop_remainder):
    name_to_features = {
        "response_input_ids": tf.FixedLenFeature([c_seq_length], tf.int64),
        "response_input_mask": tf.FixedLenFeature([c_seq_length], tf.int64),
        "response_segment_ids": tf.FixedLenFeature([c_seq_length], tf.int64),
        "response_labels": tf.FixedLenFeature([c_seq_length], tf.int64),
        "response_text_len": tf.FixedLenFeature([1], tf.int64),
        "random_forward_input_ids": tf.FixedLenFeature([a_seq_length], tf.int64),
        "random_forward_input_mask": tf.FixedLenFeature([a_seq_length], tf.int64),
        "random_forward_segment_ids": tf.FixedLenFeature([a_seq_length], tf.int64),
        "random_forward_text_len": tf.FixedLenFeature([1], tf.int64),
        "random_backward_input_ids": tf.FixedLenFeature([b_seq_length], tf.int64),
        "random_backward_input_mask": tf.FixedLenFeature([b_seq_length], tf.int64),
        "random_backward_segment_ids": tf.FixedLenFeature([b_seq_length], tf.int64),
        "random_backward_text_len": tf.FixedLenFeature([1], tf.int64),
        "random_labels": tf.FixedLenFeature([1], tf.int64),
        "swap_forward_input_ids": tf.FixedLenFeature([a_seq_length], tf.int64),
        "swap_forward_input_mask": tf.FixedLenFeature([a_seq_length], tf.int64),
        "swap_forward_segment_ids": tf.FixedLenFeature([a_seq_length], tf.int64),
        "swap_forward_text_len": tf.FixedLenFeature([1], tf.int64),
        "swap_backward_input_ids": tf.FixedLenFeature([b_seq_length], tf.int64),
        "swap_backward_input_mask": tf.FixedLenFeature([b_seq_length], tf.int64),
        "swap_backward_segment_ids": tf.FixedLenFeature([b_seq_length], tf.int64),
        "swap_backward_text_len": tf.FixedLenFeature([1], tf.int64),
        "swap_labels": tf.FixedLenFeature([1], tf.int64),
        # "generic_forward_input_ids": tf.FixedLenFeature([a_seq_length], tf.int64),
        # "generic_forward_input_mask": tf.FixedLenFeature([a_seq_length], tf.int64),
        # "generic_forward_segment_ids": tf.FixedLenFeature([a_seq_length], tf.int64),
        # "generic_forward_text_len": tf.FixedLenFeature([1], tf.int64),
        # "generic_backward_input_ids": tf.FixedLenFeature([b_seq_length], tf.int64),
        # "generic_backward_input_mask": tf.FixedLenFeature([b_seq_length], tf.int64),
        # "generic_backward_segment_ids": tf.FixedLenFeature([b_seq_length], tf.int64),
        # "generic_backward_text_len": tf.FixedLenFeature([1], tf.int64),
        # "generic_labels": tf.FixedLenFeature([1], tf.int64),
        "nli_forward_input_ids": tf.FixedLenFeature([a_seq_length], tf.int64),
        "nli_forward_input_mask": tf.FixedLenFeature([a_seq_length], tf.int64),
        "nli_forward_segment_ids": tf.FixedLenFeature([a_seq_length], tf.int64),
        "nli_forward_text_len": tf.FixedLenFeature([1], tf.int64),
        "nli_backward_input_ids": tf.FixedLenFeature([b_seq_length], tf.int64),
        "nli_backward_input_mask": tf.FixedLenFeature([b_seq_length], tf.int64),
        "nli_backward_segment_ids": tf.FixedLenFeature([b_seq_length], tf.int64),
        "nli_backward_text_len": tf.FixedLenFeature([1], tf.int64),
        "nli_labels": tf.FixedLenFeature([1], tf.int64),
        "current_utterance_id": tf.FixedLenFeature([1], tf.int64)
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       num_parallel_calls=16,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                                                       drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn


def model_fn_builder(bert_config, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps):
    def model_fn(features, labels, mode, params):
        logger.info("*** Features ***")
        for name in sorted(features.keys()):
            logger.info("  name = %s, shape = %s" % (name, features[name].shape))

        # LM
        response_input_ids = features["response_input_ids"]
        response_input_mask = features["response_input_mask"]
        response_segment_ids = features["response_segment_ids"]
        response_labels = features["response_labels"]
        response_text_len = tf.squeeze(features["response_text_len"])

        # random detection
        random_forward_input_ids = features["random_forward_input_ids"]
        random_forward_input_mask = features["random_forward_input_mask"]
        random_forward_segment_ids = features["random_forward_segment_ids"]
        random_forward_text_len = tf.squeeze(features["random_forward_text_len"])
        random_backward_input_ids = features["random_backward_input_ids"]
        random_backward_input_mask = features["random_backward_input_mask"]
        random_backward_segment_ids = features["random_backward_segment_ids"]
        random_backward_text_len = tf.squeeze(features["random_backward_text_len"])
        random_labels = features["random_labels"]

        # swap detection
        swap_forward_input_ids = features["swap_forward_input_ids"]
        swap_forward_input_mask = features["swap_forward_input_mask"]
        swap_forward_segment_ids = features["swap_forward_segment_ids"]
        swap_forward_text_len = tf.squeeze(features["swap_forward_text_len"])
        swap_backward_input_ids = features["swap_backward_input_ids"]
        swap_backward_input_mask = features["swap_backward_input_mask"]
        swap_backward_segment_ids = features["swap_backward_segment_ids"]
        swap_backward_text_len = tf.squeeze(features["swap_backward_text_len"])
        swap_labels = features["swap_labels"]

        # # generic detection
        # generic_forward_input_ids = features["generic_forward_input_ids"]
        # generic_forward_input_mask = features["generic_forward_input_mask"]
        # generic_forward_segment_ids = features["generic_forward_segment_ids"]
        # generic_forward_text_len = tf.squeeze(features["generic_forward_text_len"])
        # generic_backward_input_ids = features["generic_backward_input_ids"]
        # generic_backward_input_mask = features["generic_backward_input_mask"]
        # generic_backward_segment_ids = features["generic_backward_segment_ids"]
        # generic_backward_text_len = tf.squeeze(features["generic_backward_text_len"])
        # generic_labels = features["generic_labels"]

        # NLI detection
        nli_forward_input_ids = features["nli_forward_input_ids"]
        nli_forward_input_mask = features["nli_forward_input_mask"]
        nli_forward_segment_ids = features["nli_forward_segment_ids"]
        nli_forward_text_len = tf.squeeze(features["nli_forward_text_len"])
        nli_backward_input_ids = features["nli_backward_input_ids"]
        nli_backward_input_mask = features["nli_backward_input_mask"]
        nli_backward_segment_ids = features["nli_backward_segment_ids"]
        nli_backward_text_len = tf.squeeze(features["nli_backward_text_len"])
        nli_labels = tf.squeeze(features["nli_labels"])

        current_utterance_id = tf.squeeze(features["current_utterance_id"])

        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        logger.info("random_labels shape: {}".format(random_labels.shape))

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        random_prob, swap_prob, \
        nli_prob, \
        total_loss, lm_loss, ppl = create_bilstm_classification_model(bert_config=bert_config,
                                                                      is_training=is_training,
                                                                      response_input_ids=response_input_ids,
                                                                      response_input_mask=response_input_mask,
                                                                      response_segment_ids=response_segment_ids,
                                                                      response_text_len=response_text_len,
                                                                      response_labels=response_labels,
                                                                      random_forward_input_ids=random_forward_input_ids,
                                                                      random_forward_input_mask=random_forward_input_mask,
                                                                      random_forward_segment_ids=random_forward_segment_ids,
                                                                      random_forward_text_len=random_forward_text_len,
                                                                      random_backward_input_ids=random_backward_input_ids,
                                                                      random_backward_input_mask=random_backward_input_mask,
                                                                      random_backward_segment_ids=random_backward_segment_ids,
                                                                      random_backward_text_len=random_backward_text_len,
                                                                      random_labels=random_labels,
                                                                      swap_forward_input_ids=swap_forward_input_ids,
                                                                      swap_forward_input_mask=swap_forward_input_mask,
                                                                      swap_forward_segment_ids=swap_forward_segment_ids,
                                                                      swap_forward_text_len=swap_forward_text_len,
                                                                      swap_backward_input_ids=swap_backward_input_ids,
                                                                      swap_backward_input_mask=swap_backward_input_mask,
                                                                      swap_backward_segment_ids=swap_backward_segment_ids,
                                                                      swap_backward_text_len=swap_backward_text_len,
                                                                      swap_labels=swap_labels,
                                                                      # generic_forward_input_ids=generic_forward_input_ids,
                                                                      # generic_forward_input_mask=generic_forward_input_mask,
                                                                      # generic_forward_segment_ids=generic_forward_segment_ids,
                                                                      # generic_forward_text_len=generic_forward_text_len,
                                                                      # generic_backward_input_ids=generic_backward_input_ids,
                                                                      # generic_backward_input_mask=generic_backward_input_mask,
                                                                      # generic_backward_segment_ids=generic_backward_segment_ids,
                                                                      # generic_backward_text_len=generic_backward_text_len,
                                                                      # generic_labels=generic_labels,
                                                                      nli_forward_input_ids=nli_forward_input_ids,
                                                                      nli_forward_input_mask=nli_forward_input_mask,
                                                                      nli_forward_segment_ids=nli_forward_segment_ids,
                                                                      nli_forward_text_len=nli_forward_text_len,
                                                                      nli_backward_input_ids=nli_backward_input_ids,
                                                                      nli_backward_input_mask=nli_backward_input_mask,
                                                                      nli_backward_segment_ids=nli_backward_segment_ids,
                                                                      nli_backward_text_len=nli_backward_text_len,
                                                                      nli_labels=nli_labels,
                                                                      num_nli_labels=3,
                                                                      use_one_hot_embeddings=False,
                                                                      l2_reg_lambda=FLAGS.l2_reg_lambda,
                                                                      dropout_rate=FLAGS.dropout_rate,
                                                                      lstm_size=FLAGS.lstm_size,
                                                                      num_layers=FLAGS.num_layers)

        tvars = tf.trainable_variables()

        # 加载ROBERTa模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # 打印变量名
        logger.info("**** Trainable Variables ****")

        # 打印加载模型的参数
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logger.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            # train_op = optimizer.optimizer(total_loss, learning_rate, num_train_steps)

            train_op = optimization.create_optimizer(total_loss * 5 + lm_loss,
                                                     learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps,
                                                     False)
            hook_dict = {}
            hook_dict['discriminative_loss'] = total_loss * 5
            hook_dict['lm_loss'] = lm_loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=100)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss*5 + lm_loss,
                train_op=train_op,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(random_labels, swap_labels, nli_labels,
                          random_prob, swap_prob, nli_prob):

                nli_prec = precision(labels=nli_labels,
                                     predictions=tf.argmax(nli_prob, axis=1),
                                     num_classes=3,
                                     pos_indices=[2])

                nli_rec = recall(labels=nli_labels,
                                 predictions=tf.argmax(nli_prob, axis=1),
                                 num_classes=3,
                                 pos_indices=[2])

                nli_fscore = f1(labels=nli_labels,
                                predictions=tf.argmax(nli_prob, axis=1),
                                num_classes=3, pos_indices=[2])

                return {
                    "nli-precision": nli_prec,
                    "nli-recall": nli_rec,
                    "nli-f1-score": nli_fscore,
                    "random-category-accuracy": tf.metrics.accuracy(labels=random_labels,
                                                                    predictions=tf.argmax(random_prob, axis=1)),
                    # "generic-category-accuracy": tf.metrics.accuracy(labels=generic_labels,
                    #                                                  predictions=tf.argmax(generic_prob, axis=1)),
                    "swap-category-accuracy": tf.metrics.accuracy(labels=swap_labels,
                                                                  predictions=tf.argmax(swap_prob, axis=1))
                }

            eval_metrics = metric_fn(random_labels, swap_labels, nli_labels, random_prob, swap_prob, nli_prob)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss * 5 + lm_loss,
                eval_metric_ops=eval_metrics
            )

        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"random_prob": random_prob,
                             # "generic_prob": generic_prob,
                             "swap_prob": swap_prob,
                             "nli_prob": nli_prob,
                             "perplexity": ppl,
                             "utterance_ids": current_utterance_id}
            )
        return output_spec

    return model_fn


def get_last_checkpoint(model_path):
    if not os.path.exists(os.path.join(model_path, 'checkpoint')):
        logger.info('checkpoint file not exits:'.format(os.path.join(model_path, 'checkpoint')))
        return None
    last = None
    with codecs.open(os.path.join(model_path, 'checkpoint'), 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip().split(':')
            if len(line) != 2:
                continue
            if line[0] == 'model_checkpoint_path':
                last = line[1][2:-1]
                break
    return last


def adam_filter(model_path):
    """
    去掉模型中的Adam相关参数，这些参数在测试的时候是没有用的
    :param model_path:
    :return:
    """
    last_name = get_last_checkpoint(model_path)
    if last_name is None:
        return
    sess = tf.Session()
    imported_meta = tf.train.import_meta_graph(os.path.join(model_path, last_name + '.meta'))
    imported_meta.restore(sess, os.path.join(model_path, last_name))
    need_vars = []
    for var in tf.global_variables():
        if 'adam_v' not in var.name and 'adam_m' not in var.name:
            need_vars.append(var)
    saver = tf.train.Saver(need_vars)
    saver.save(sess, os.path.join(model_path, 'model.ckpt'))


def main(_):
    processors = {
        "nsp": NspProcessor
    }

    roberta_config = modeling.BertConfig.from_json_file(FLAGS.roberta_config_file)

    if FLAGS.clean and FLAGS.do_train:
        if os.path.exists(FLAGS.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                del_file(FLAGS.output_dir)
            except Exception as e:
                logger.info(e)
                logger.info('please remove the files of output dir and data.conf')
                exit(-1)

    # check output dir exists
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    processor = processors[FLAGS.task](FLAGS.output_dir)

    logger.info("total vocabulary size is: {}".format(roberta_config.vocab_size))

    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        session_config=session_config,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max
    )

    train_examples = None
    valid_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train and FLAGS.do_eval:
        if not os.path.exists(os.path.join(FLAGS.output_dir, 'train_lines.pkl')):
            train_dialogues, train_dialogues_uids = processor.process_dialogue(FLAGS.data_dir, FLAGS.corpus_name,
                                                                               'train')
        else:
            train_dialogues, train_dialogues_uids = None, None

        train_examples = processor.get_train_example(train_dialogues,
                                                     train_dialogues_uids,
                                                     context_window_size=FLAGS.window_size,
                                                     split='train')
        num_train_steps = int(
            len(train_examples) * 1.0 / FLAGS.batch_size * FLAGS.num_train_epochs)
        if num_train_steps < 1:
            raise AttributeError('training data is so small...')
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", FLAGS.batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        if not os.path.exists(os.path.join(FLAGS.output_dir, 'valid_lines.pkl')):
            valid_dialogues, valid_dialogues_uids = processor.process_dialogue(FLAGS.data_dir, FLAGS.corpus_name,
                                                                               'valid')
        else:
            valid_dialogues, valid_dialogues_uids = None, None

        valid_examples = processor.get_valid_example(valid_dialogues,
                                                     valid_dialogues_uids,
                                                     context_window_size=FLAGS.window_size,
                                                     split='valid')

        num_valid_steps = int(len(valid_examples) * 1.0 / FLAGS.batch_size)

        # 打印验证集数据信息
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(valid_examples))
        logger.info("  Batch size = %d", FLAGS.batch_size)
        logger.info("  Num valid steps = %d", num_valid_steps)

    train_utterance_label_list = processor.get_eval_utterance_labels(FLAGS.data_dir, FLAGS.corpus_name)

    train_utterance_label_map = {}
    for (i, label) in enumerate(train_utterance_label_list, 1):
        train_utterance_label_map[label] = i

    if not os.path.exists(os.path.join(FLAGS.output_dir, 'train_utterance_label2id.pkl')):
        with codecs.open(os.path.join(FLAGS.output_dir, 'train_utterance_label2id.pkl'), 'wb') as w:
            pickle.dump(train_utterance_label_map, w)

    # 返回的model_dn 是一个函数，其定义了模型，训练，评测方法，并且使用钩子参数，加载了BERT模型的参数进行了自己模型的参数初始化过程
    # tf 新的架构方法，通过定义model_fn 函数，定义模型，然后通过EstimatorAPI进行模型的其他工作，Es就可以控制模型的训练，预测，评估工作等。
    model_fn = model_fn_builder(
        bert_config=roberta_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    params = {
        'batch_size': FLAGS.batch_size
    }

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    if FLAGS.do_train and FLAGS.do_eval:
        # 1. 将数据转化为tf_record 数据
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            filed_based_convert_examples_to_features(
                train_examples, train_utterance_label_map, FLAGS.max_pre_len,
                FLAGS.max_post_len, FLAGS.max_seq_len, train_file, is_training=True)

        # 2.读取record 数据，组成batch
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            a_seq_length=FLAGS.max_pre_len,
            b_seq_length=FLAGS.max_post_len,
            c_seq_length=FLAGS.max_seq_len,
            is_training=True,
            drop_remainder=True)
        # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        valid_file = os.path.join(FLAGS.output_dir, "valid.tf_record")
        if not os.path.exists(valid_file):
            filed_based_convert_examples_to_features(
                valid_examples, train_utterance_label_map,
                FLAGS.max_pre_len, FLAGS.max_post_len, FLAGS.max_seq_len,
                valid_file, is_training=True)

        valid_input_fn = file_based_input_fn_builder(
            input_file=valid_file,
            a_seq_length=FLAGS.max_pre_len,
            b_seq_length=FLAGS.max_post_len,
            c_seq_length=FLAGS.max_seq_len,
            is_training=False,
            drop_remainder=True)

        # train and eval together
        # early stop hook
        early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name='loss',
            max_steps_without_decrease=num_train_steps,
            eval_dir=None,
            min_steps=0,
            run_every_secs=None,
            run_every_steps=FLAGS.save_checkpoints_steps)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps,
                                            hooks=[early_stopping_hook])

        best_copier = BestCheckpointCopier(
            name='best',  # directory within model directory to copy checkpoints to
            checkpoints_to_keep=1,  # number of checkpoints to keep
            score_metric='nli-f1-score',  # metric to use to determine "best"
            compare_fn=lambda x, y: x.score > y.score,
            sort_key_fn=lambda x: x.score,
            sort_reverse=True)  # sort order when discarding excess checkpoints

        eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, steps=FLAGS.valid_steps, exporters=best_copier)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if FLAGS.do_predict:

        predict_utterance_label_list = processor.get_eval_utterance_labels(FLAGS.data_dir, FLAGS.corpus_name)

        predict_utterance_label_map = {}
        for (i, label) in enumerate(predict_utterance_label_list, 1):
            predict_utterance_label_map[label] = i

        predict_utterance_id2label = {v: k for k, v in predict_utterance_label_map.items()}

        if not os.path.exists(os.path.join(FLAGS.output_dir, '{}_lines.pkl'.format(FLAGS.eval_type))):
            eval_dialogues, eval_dialogues_uids = processor.process_dialogue(FLAGS.data_dir, FLAGS.corpus_name,
                                                                             FLAGS.eval_type)
        else:
            eval_dialogues, eval_dialogues_uids = None, None

        predict_examples = processor.get_eval_example(eval_dialogues,
                                                      eval_dialogues_uids,
                                                      context_window_size=FLAGS.window_size,
                                                      split=FLAGS.eval_type)

        predict_file = os.path.join(FLAGS.output_dir, "{}.tf_record".format(FLAGS.eval_type))

        if not os.path.exists(predict_file):
            filed_based_convert_examples_to_features(predict_examples, predict_utterance_label_map,
                                                     FLAGS.max_pre_len, FLAGS.max_post_len, FLAGS.max_seq_len,
                                                     predict_file, is_training=False)

        logger.info("***** Running prediction*****")
        logger.info("  Num examples = %d", len(predict_examples))
        logger.info("  Batch size = %d", FLAGS.batch_size)

        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            a_seq_length=FLAGS.max_pre_len,
            b_seq_length=FLAGS.max_post_len,
            c_seq_length=FLAGS.max_seq_len,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir,
                                           "{}_{}_confidence_scores.txt".format(FLAGS.corpus_name, FLAGS.eval_type))

        def result_to_pair(write_agent):
            for predict_line, prediction in zip(predict_examples, result):
                line = ''
                try:
                    line += str(predict_utterance_id2label[prediction['utterance_ids']]) + '\t' + \
                            str(prediction['random_prob'][1]) + '\t' + \
                            str(prediction['swap_prob'][1]) + '\t' + \
                            str(prediction['perplexity']) + '\t' + \
                            '\t'.join([str(item) for item in prediction['nli_prob']]) + '\n'
                except Exception as e:
                    logger.info(e)
                    break
                write_agent.write(line)

        with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
            result_to_pair(writer)

    # filter model
    if FLAGS.filter_adam_var:
        adam_filter(FLAGS.output_dir)


if __name__ == "__main__":
    tf.app.run()
