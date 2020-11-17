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
import torch
import codecs
import pickle
import pandas as pd
import random
import argparse
import logging
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from absl import app
from transformers import RobertaTokenizer
from transformers import RobertaModel, RobertaConfig
from torch.utils.data import Dataset, DataLoader

from unified_framework import InputFeatures, InputExample
from modeling import DscoreModel

random.seed(2000)

bert_path = '../pretrained_module/roberta/roberta-base-pretrained-tf'

root_path = '/home/chen/hade_main'

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument('--data_dir', default=os.path.join(root_path, 'datasets/persona-chat/'),
                    help="train, dev and test data dir", type=str)

parser.add_argument('--roberta_config_file', default=os.path.join(bert_path, 'roberta-base-config.json'),
                    help='roberta config file path', type=str)

parser.add_argument('--output_dir', default=os.path.join(bert_path, 'roberta_base.ckpt'),
                    help='directory to store trained model', type=str)

parser.add_argument('--init_checkpoint', default=None,
                    help='Initial checkpoint (usually from a pre-trained model).', type=str)

parser.add_argument('--corpus_name', default='book', help="corpus name", type=str)

parser.add_argument('--task', default='nsp', help='which model to train', type=str)

parser.add_argument('--eval_type', default='eval', help='specify type of data to evaluate', type=str)

parser.add_argument('--dupe_factor', default=1, help='number of times to duplicate input', type=int)

parser.add_argument('max_pre_len', default=256,
                    help='The maximum total input sequence length after Sentencepiece tokenization.',
                    type=int)

parser.add_argument('max_post_len', default=256,
                    help='The maximum total input sequence length after Sentencepiece tokenization.',
                    type=int)

parser.add_argument('max_seq_len', default=256,
                    help='The maximum total response sequence length after Sentencepiece tokenization.',
                    type=int)

parser.add_argument('window_size', default=5, help='number of utterances in a context window', type=int)

parser.add_argument('batch_size', default=32, help='Total batch size for training, eval and predict.', type=int)

parser.add_argument('num_train_epochs', default=10, help='Total number of training epochs to perform.', type=int)

parser.add_argument('embedding_dim', default=768, help='embedding dimension.', type=int)

parser.add_argument('lstm_size', default=300, help='size of lstm units.', type=int)

parser.add_argument('num_layers', default=1, help='number of rnn layers, default is 1.', type=int)

parser.add_argument('keep_checkpoint_max', default=3,
                    help='number of checkpoints to keep, default is 3.', type=int)

parser.add_argument('learning_rate', default=1e-5, help='The initial learning rate for Adam.', type=float)

parser.add_argument('dropout_rate', default=0.5, help='Dropout rate', type=float)

parser.add_argument('l2_reg_lambda', default=0.2, help='l2_reg_lambda', type=float)

parser.add_argument('warmup_proportion', default=0.025,
                    help='Proportion of training to perform linear learning rate warmup for '
                         'E.g., 0.1 = 10% of training.', type=float)

parser.add_argument('do_train', default=False, help='Whether to run training.', type=bool)

parser.add_argument('do_eval', default=False,
                    help='Whether to run eval on the dev set.', type=bool)

parser.add_argument('do_predict', default=False,
                    help='Whether to run the predict in inference mode on the test set.', type=bool)

parser.add_argument('filter_adam_var', default=False,
                    help='after training do filter Adam params from model and save no Adam params model in file.',
                    type=bool)

parser.add_argument('do_lower_case', default=True, help='Whether to lower case the input text.', type=bool)

parser.add_argument('clean', default=False, help="whether to clean output folder", type=bool)

opt = parser.parse_args()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

logger = logging.getLogger('[D-score]')


class Randomizer(object):

    def __init__(self, output_dir):
        self.labels = []
        self.output_dir = output_dir
        with codecs.open('dull_responses.txt', mode='r', encoding='utf-8') as rf:
            lines = rf.readlines()
        self.dull_responses = [l.strip() for l in lines]

    def process_data(self, data_dir, corpus_name, split):
        logger.info("*************** reading {} data*****************************".format(split))
        data = pd.read_csv(os.path.join(data_dir, corpus_name + '_main.csv'))
        meta_data = pd.read_csv(os.path.join(data_dir, corpus_name + '_metadata.csv'))
        paragraph_ids = list(meta_data[meta_data['type'] == split]['paragraph_id'])
        paragrahs = []
        paragraph_uids = []
        for i in tqdm(paragraph_ids):
            single_paragraph = [str(item) for item in list(data[data['UID'].str.startswith(i)]['SEG'])]
            single_paragraph_uid = list(data[data['UID'].str.startswith(i)]['UID'])
            single_paragraph.insert(0, 'start of paragraph')
            single_paragraph_uid.insert(0, corpus_name + '-' + 'start-placeholder')
            single_paragraph.append('end of paragraph')
            single_paragraph_uid.append(corpus_name + '-' + 'end-placeholder')
            paragrahs.append(single_paragraph)
            paragraph_uids.append(single_paragraph_uid)
        return paragrahs, paragraph_uids

    def get_labels(self):
        self.labels.append('original')
        self.labels.append('swap')
        self.labels.append('random')
        return self.labels

    def get_sentence_ids(self, data_dir, corpus_name):
        data = pd.read_csv(os.path.join(data_dir, corpus_name + '_main.csv'))
        ids = list(data['UID'])
        return ids

    def get_train_example(self, train_paragraphs, train_paragraph_uids, context_window_size, split='train'):
        if os.path.exists(os.path.join(opt.output_dir, '{}_lines.pkl'.format(split))):
            return self._create_example(pickle.load(open(os.path.join(opt.output_dir,
                                                                      '{}_lines.pkl'.format(split)), 'rb')), split)
        else:
            lines = self._read_data(train_paragraphs, train_paragraph_uids, context_window_size, split)
            pickle.dump(lines, open(os.path.join(opt.output_dir, 'train_lines.pkl'), 'wb'))
            return self._create_example(lines, split)

    def get_valid_example(self, valid_paragraphs, valid_paragraph_uids, context_window_size, split='valid'):
        if os.path.exists(os.path.join(opt.output_dir, '{}_lines.pkl'.format(split))):
            return self._create_example(pickle.load(open(os.path.join(opt.output_dir,
                                                                      'valid_lines.pkl'), 'rb')), split)
        else:
            lines = self._read_data(valid_paragraphs, valid_paragraph_uids, context_window_size, split)
            pickle.dump(lines, open(os.path.join(opt.output_dir, '{}_lines.pkl'.format(split)), 'wb'))
            return self._create_example(lines, split)

    def get_eval_example(self, eval_paragraphs, eval_paragraph_uids, context_window_size, split):
        if os.path.exists(os.path.join(opt.output_dir, '{}_lines.pkl'.format(split))):
            return self._create_example(pickle.load(open(os.path.join(opt.output_dir,
                                                                      '{}_lines.pkl'.format(split)), 'rb')), split)
        else:
            lines = self._read_data(eval_paragraphs, eval_paragraph_uids, context_window_size, split)
            pickle.dump(lines, open(os.path.join(opt.output_dir, '{}_lines.pkl'.format(split)), 'wb'))
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
            random_utt = line[6]
            swap_pre = line[7]
            rand_current = line[8]
            # if i == 0:
            #     logger.info('label: ', label)
            examples.append(InputExample(guid=guid, utterance_id=utterance_id,
                                         correct_pre_context=correct_pre_context,
                                         correct_post_context=correct_post_context,
                                         correct_current_response=correct_current_response,
                                         swapped_context=swapped_context,
                                         swapped_current_response=swapped_current_response,
                                         random_utt=random_utt,
                                         swap_pre=swap_pre,
                                         rand_current=rand_current))
        return examples

    def _read_data(self, paragraphs, paragraph_uids, context_window_size, split):
        if context_window_size <= 2:
            raise ValueError("window size must be larger than 2")
        logger.info("*************** form {} sentence triplet *****************************".format(split))
        lines = []
        if split == 'train' or split == 'valid':
            for t in range(opt.dupe_factor):
                for idx, d in enumerate(tqdm(paragraphs)):
                    if len(d) < context_window_size:
                        raise ValueError("length of paragraph {0} is less than "
                                         "window size of {1}".format(idx, context_window_size))
                    elif len(d) == context_window_size:
                        for i in range(1, len(d) - 1, 1):
                            current_utt = d[i]
                            pre_context = d[:i]
                            post_context = d[i + 1:]
                            utterance_idx = paragraph_uids[idx][i]
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

                            rand_num = random.random()
                            if rand_num < 0.5:
                                # random choice from other dialogues
                                random_para_idx = random.choice(list(range(0, len(paragraphs), 1)))
                                while random_para_idx == idx:
                                    random_para_idx = random.choice(list(range(0, len(paragraphs), 1)))
                                random_utt = random.choice(paragraphs[random_para_idx])
                                while random_utt == current_utt:
                                    random_utt = random.choice(paragraphs[random_para_idx])
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
                                          random_utt, swap_pre, rand_current])
                    else:
                        for j in range(0, len(d) - context_window_size + 1, 1):
                            window_sized_d = d[j:j + context_window_size]
                            for k in range(1, len(window_sized_d) - 1, 1):
                                current_utt = window_sized_d[k]
                                pre_context = window_sized_d[:k]
                                post_context = window_sized_d[k + 1:]
                                utterance_idx = paragraph_uids[idx][j + k]
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

                                # random choice from other dialogues
                                rand_num = random.random()
                                if rand_num < 0.5:
                                    random_para_idx = random.choice(list(range(0, len(paragraphs), 1)))
                                    while random_para_idx == idx:
                                        random_para_idx = random.choice(list(range(0, len(paragraphs), 1)))
                                    random_utt = random.choice(paragraphs[random_para_idx])
                                    while random_utt == current_utt:
                                        random_utt = random.choice(paragraphs[random_para_idx])
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
                                              random_utt, swap_pre, rand_current])
        else:
            for idx, d in enumerate(tqdm(paragraphs)):
                if len(d) < context_window_size:
                    raise ValueError("length of dialogue {0} is less "
                                     "than window size of {1}".format(idx, context_window_size))
                elif len(d) == context_window_size:
                    for i in range(1, len(d) - 1, 1):
                        current_utt = d[i]
                        pre_context = d[:i]
                        post_context = d[i + 1:]

                        utterance_idx = paragraph_uids[idx][i]
                        correct_pre_context = ' </s> '.join(pre_context)
                        correct_post_context = ' </s> '.join(post_context)
                        correct_current_response = current_utt

                        lines.append([utterance_idx, correct_pre_context, correct_post_context,
                                      correct_current_response, correct_pre_context, correct_current_response,
                                      correct_current_response, False, False])

                else:
                    for j in range(0, len(d) - context_window_size + 1, 1):
                        window_sized_d = d[j:j + context_window_size]
                        for k in range(1, len(window_sized_d) - 1, 1):
                            current_utt = window_sized_d[k]
                            pre_context = window_sized_d[:k]
                            post_context = window_sized_d[k + 1:]
                            utterance_idx = paragraph_uids[idx][j + k]
                            correct_pre_context = ' </s> '.join(pre_context)
                            correct_post_context = ' </s> '.join(post_context)
                            correct_current_response = current_utt

                            lines.append([utterance_idx, correct_pre_context, correct_post_context,
                                          correct_current_response, correct_pre_context, correct_current_response,
                                          correct_current_response, False, False])
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


class DscoreDataset(Dataset):
    def __init__(self, features):
        """
        Args:
            features (list of feature objects): the dataset
        """
        self._features = features
        self._target_size = len(features)

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dict of the data point's features (x_data) and label (y_target)
        """
        row = self._features[index]
        response_input_ids = row.response_input_ids
        response_input_mask = row.response_input_mask
        response_segment_ids = row.response_segment_ids
        response_labels = row.response_labels
        response_text_len = row.response_text_len
        random_forward_input_ids = row.random_forward_input_ids
        random_forward_input_mask = row.random_forward_input_mask
        random_forward_segment_ids = row.random_forward_segment_ids
        random_forward_text_len = row.random_forward_text_len
        random_backward_input_ids = row.random_backward_input_ids
        random_backward_input_mask = row.random_backward_input_mask
        random_backward_segment_ids = row.random_backward_segment_ids
        random_backward_text_len = row.random_backward_text_len
        random_labels = row.random_labels
        swap_forward_input_ids = row.swap_forward_input_ids
        swap_forward_input_mask = row.swap_forward_input_mask
        swap_forward_segment_ids = row.swap_forward_segment_ids
        swap_forward_text_len = row.swap_forward_text_len
        swap_backward_input_ids = row.swap_backward_input_ids
        swap_backward_input_mask = row.swap_backward_input_mask
        swap_backward_segment_ids = row.swap_backward_segment_ids
        swap_backward_text_len = row.swap_backward_text_len
        swap_labels = row.swap_labels
        current_utterance_id = row.current_utterance_id

        return {'response_input_ids': response_input_ids,
                'response_input_mask': response_input_mask,
                'response_segment_ids': response_segment_ids,
                'response_labels': response_labels,
                'response_text_len': response_text_len,
                'random_forward_input_ids': random_forward_input_ids,
                'random_forward_input_mask': random_forward_input_mask,
                'random_forward_segment_ids': random_forward_segment_ids,
                'random_forward_text_len': random_forward_text_len,
                'random_backward_input_ids': random_backward_input_ids,
                'random_backward_input_mask': random_backward_input_mask,
                'random_backward_segment_ids': random_backward_segment_ids,
                'random_backward_text_len': random_backward_text_len,
                'random_labels': random_labels,
                'swap_forward_input_ids': swap_forward_input_ids,
                'swap_forward_input_mask': swap_forward_input_mask,
                'swap_forward_segment_ids': swap_forward_segment_ids,
                'swap_forward_text_len': swap_forward_text_len,
                'swap_backward_input_ids': swap_backward_input_ids,
                'swap_backward_input_mask': swap_backward_input_mask,
                'swap_backward_segment_ids': swap_backward_segment_ids,
                'swap_backward_text_len': swap_backward_text_len,
                'swap_labels': swap_labels,
                'current_utterance_id': current_utterance_id}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


def prepare_data(examples,
                 utterance_label_map,
                 max_pre_len,
                 max_post_len,
                 max_seq_len,
                 tokenizer=None,
                 is_training=True):

    feature_list = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Reading example %d of %d" % (ex_index, len(examples)))

        feature = prepare_single_example(ex_index, example, utterance_label_map,
                                         max_pre_len, max_post_len, max_seq_len,
                                         tokenizer=tokenizer, is_training=is_training)
        feature_list.append(feature)

    return DscoreDataset(feature_list)


def prepare_single_example(ex_index, example, utterance_label_map, max_pre_len, max_post_len, max_seq_len,
                           tokenizer=None, is_training=True):
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
    org_response_input_ids = tokenizer.encode(' '.join([example.correct_pre_context,
                                                        example.correct_current_response,
                                                        example.correct_post_context]))

    _truncate_seq_pair_back(org_response_input_ids, max_seq_len)

    response_input_ids = org_response_input_ids[:-1]
    response_labels = org_response_input_ids[1:]
    response_input_mask = [1] * len(response_input_ids)
    response_segment_ids = [1] * len(response_input_ids)
    response_text_len = len(response_input_ids)

    # Zero-pad up to the sequence length.
    while len(response_input_ids) < max_pre_len + max_post_len:
        # pad token id is 1
        response_input_ids.append(1)
        response_input_mask.append(0)
        response_segment_ids.append(0)
        response_labels.append(1)

    assert len(response_input_ids) == max_pre_len + max_post_len
    assert len(response_input_mask) == max_pre_len + max_post_len
    assert len(response_segment_ids) == max_pre_len + max_post_len
    assert len(response_labels) == max_pre_len + max_post_len

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

    # 结构化为一个类
    feature = InputFeatures(
        response_input_ids=np.array(response_input_ids, dtype=np.int64),
        response_input_mask=np.array(response_input_mask, dtype=np.int64),
        response_segment_ids=np.array(response_segment_ids, dtype=np.int64),
        response_labels=np.array(response_labels, dtype=np.float32),
        response_text_len=np.array(response_text_len, dtype=np.float32),
        random_forward_input_ids=np.array(random_forward_input_ids, dtype=np.int64),
        random_forward_input_mask=np.array(random_forward_input_mask, dtype=np.int64),
        random_forward_segment_ids=np.array(random_forward_segment_ids, dtype=np.int64),
        random_forward_text_len=np.array(random_forward_text_len, dtype=np.float32),
        random_backward_input_ids=np.array(random_backward_input_ids, dtype=np.int64),
        random_backward_input_mask=np.array(random_backward_input_mask, dtype=np.int64),
        random_backward_segment_ids=np.array(random_backward_segment_ids, dtype=np.int64),
        random_backward_text_len=np.array(random_backward_text_len, dtype=np.float32),
        random_labels=np.array(random_labels, dtype=np.float32),
        swap_forward_input_ids=np.array(swap_forward_input_ids, dtype=np.int64),
        swap_forward_input_mask=np.array(swap_forward_input_mask, dtype=np.int64),
        swap_forward_segment_ids=np.array(swap_forward_segment_ids, dtype=np.int64),
        swap_forward_text_len=np.array(swap_forward_text_len, dtype=np.float32),
        swap_backward_input_ids=np.array(swap_backward_input_ids, dtype=np.int64),
        swap_backward_input_mask=np.array(swap_backward_input_mask, dtype=np.int64),
        swap_backward_segment_ids=np.array(swap_backward_segment_ids, dtype=np.int64),
        swap_backward_text_len=np.array(swap_backward_text_len, dtype=np.float32),
        swap_labels=np.array(swap_labels, dtype=np.float32),
        current_utterance_id=np.array(current_utterance_id, dtype=np.float32)
    )

    return feature


def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


def make_train_state():
    return {'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1}


def main(_):

    processors = {
        "nsp": Randomizer
    }

    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    roberta_model = RobertaModel.from_pretrained(opt.init_checkpoint)

    if opt.clean and opt.do_train:
        if os.path.exists(opt.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                del_file(opt.output_dir)
            except Exception as e:
                logger.info(e)
                logger.info('please remove the files of output dir and data.conf')
                exit(-1)

    # check output dir exists
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    processor = processors[opt.task](opt.output_dir)

    logger.info("total vocabulary size is: {}".format(roberta_tokenizer.vocab_size))

    train_examples = None
    valid_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if opt.do_train and opt.do_eval:
        if not os.path.exists(os.path.join(opt.output_dir, 'train_lines.pkl')):
            train_paragraphs, train_paragraph_uids = processor.process_data(opt.data_dir, opt.corpus_name, 'train')
        else:
            train_paragraphs, train_paragraph_uids = None, None

        train_examples = processor.get_train_example(train_paragraphs,
                                                     train_paragraph_uids,
                                                     context_window_size=opt.window_size,
                                                     split='train')
        num_train_steps = int(
            len(train_examples) * 1.0 / opt.batch_size * opt.num_train_epochs)
        if num_train_steps < 1:
            raise AttributeError('training data is so small...')

        num_warmup_steps = int(num_train_steps * opt.warmup_proportion)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", opt.batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        if not os.path.exists(os.path.join(opt.output_dir, 'valid_lines.pkl')):
            valid_paragraphs, valid_paragraph_uids = processor.process_data(opt.data_dir, opt.corpus_name, 'valid')
        else:
            valid_paragraphs, valid_paragraph_uids = None, None

        valid_examples = processor.get_valid_example(valid_paragraphs,
                                                     valid_paragraph_uids,
                                                     context_window_size=opt.window_size,
                                                     split='valid')

        num_valid_steps = int(len(valid_examples) * 1.0 / opt.batch_size)

        # 打印验证集数据信息
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(valid_examples))
        logger.info("  Batch size = %d", opt.batch_size)
        logger.info("  Num valid steps = %d", num_valid_steps)

    utterance_label_list = processor.get_sentence_ids(opt.data_dir, opt.corpus_name)

    utterance_label_map = {}
    for (i, label) in enumerate(utterance_label_list, 1):
        utterance_label_map[label] = i

    logger.info("Initializing D-score model --------------------------------------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DscoreModel(roberta_model, opt.embedding_dim, opt.lstm_size,
                        opt.dropout_rate, opt.num_layers, device, roberta_tokenizer.vocab_size,
                        num_random_classes=2, num_swap_classes=2, is_training=opt.do_train, bidirectional=True)

    if opt.do_train and opt.do_eval:
        train_dataset = prepare_data(train_examples, utterance_label_map, opt.max_pre_len,
                                     opt.max_post_len, opt.max_seq_len, tokenizer=roberta_tokenizer, is_training=True)
        valid_dataset = prepare_data(valid_examples, utterance_label_map, opt.max_pre_len,
                                     opt.max_post_len, opt.max_seq_len, tokenizer=roberta_tokenizer, is_training=True)

        train_state = make_train_state()

        # loss and optimizer
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

        epoch = 0
        best_epoch = 0
        max_acc = 10
        torch.save(model.state_dict(), os.path.join(opt.output_dir, 'Dscore-{:04d}.pt'.format(epoch)))
        for epoch_index in range(opt.num_train_epochs):
            train_state['epoch_index'] = epoch_index

            # Iterate over training dataset

            # setup: batch generator, set loss and acc to 0, set train mode on
            batch_generator = generate_batches(train_dataset,
                                               batch_size=opt.batch_size,
                                               device=device)
            running_loss = 0.0
            running_acc = 0.0

            model.train()

            # return {'response_input_ids': response_input_ids,
            #         'response_input_mask': response_input_mask,
            #         'response_segment_ids': response_segment_ids,
            #         'response_labels': response_labels,
            #         'response_text_len': response_text_len,
            #         'random_forward_input_ids': random_forward_input_ids,
            #         'random_forward_input_mask': random_forward_input_mask,
            #         'random_forward_segment_ids': random_forward_segment_ids,
            #         'random_forward_text_len': random_forward_text_len,
            #         'random_backward_input_ids': random_backward_input_ids,
            #         'random_backward_input_mask': random_backward_input_mask,
            #         'random_backward_segment_ids': random_backward_segment_ids,
            #         'random_backward_text_len': random_backward_text_len,
            #         'random_labels': random_labels,
            #         'swap_forward_input_ids': swap_forward_input_ids,
            #         'swap_forward_input_mask': swap_forward_input_mask,
            #         'swap_forward_segment_ids': swap_forward_segment_ids,
            #         'swap_forward_text_len': swap_forward_text_len,
            #         'swap_backward_input_ids': swap_backward_input_ids,
            #         'swap_backward_input_mask': swap_backward_input_mask,
            #         'swap_backward_segment_ids': swap_backward_segment_ids,
            #         'swap_backward_text_len': swap_backward_text_len,
            #         'swap_labels': swap_labels,
            #         'current_utterance_id': current_utterance_id}

            for batch_index, batch_dict in enumerate(batch_generator):
                # the training routine is 5 steps:

                # step 1. zero the gradients
                optimizer.zero_grad()

                # step 2. compute the output
                random_preds, random_prob, swap_preds, swap_prob, response_outputs = model(feauture=batch_dict,
                                                                                           batch_size=opt.batch_size)

                # step 3. compute the loss
                random_loss = loss_func(random_preds, batch_dict["random_labels"])
                swap_loss = loss_func(swap_preds, batch_dict["swap_labels"])
                lm_loss = loss_func(response_outputs, batch_dict["response_labels"])

                loss_batch = loss.item()

                running_loss += (loss_batch - running_loss) / (batch_index + 1)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()

                # -----------------------------------------
                # compute the accuracy
                acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_batch - running_acc) / (batch_index + 1)

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0, set eval mode on
            dataset.set_split('val')
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.
            running_acc = 0.
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                # step 1. compute the output
                y_pred = classifier(x_in=batch_dict['x_data'].float())

                # step 2. compute the loss
                loss = loss_func(y_pred, batch_dict['y_target'].float())
                loss_batch = loss.item()
                running_loss += (loss_batch - running_loss) / (batch_index + 1)

                # step 3. compute the accuracy
                acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
                running_acc += (acc_batch - running_acc) / (batch_index + 1)

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)


        x_lm_sequence_segment = feature.response_segment_ids
        x_lm_sequence_labels = feature.response_labels

        x_lm_sequence_text_len = feature.response_text_len

        y_random = feature.random_labels

        random_one_hot = F.one_hot(y_random, num_classes=self.num_random_classes)

        random_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=random_one_hot,
                                                                             logits=random_preds))
        random_prob = tf.nn.softmax(random_preds)

        y_swap = feature.swap_labels
        current_utterance_id = feature.current_utterance_id

        response_one_hot = tf.one_hot(response_labels, depth=config.vocab_size, dtype=tf.float32)

        lm_cost = tf.nn.softmax_cross_entropy_with_logits(labels=response_one_hot, logits=response_outputs)

        sequence_mask = tf.sequence_mask(response_text_len, maxlen=response_embedding_shape[1], dtype=tf.float32)

        masked_lm_cost = tf.math.multiply(lm_cost, sequence_mask)

        final_lm_loss = tf.reduce_mean(
            tf.math.divide(tf.reduce_sum(masked_lm_cost, axis=1), tf.cast(response_text_len, dtype=tf.float32)))

        perplexity = tf.exp(
            tf.math.divide(tf.reduce_sum(masked_lm_cost, axis=1), tf.cast(response_text_len, dtype=tf.float32)))


if __name__ == "__main__":
    app.run(main)
