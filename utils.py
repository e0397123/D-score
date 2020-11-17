"""
@Author:Zhang Chen
"""


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, utterance_id, correct_pre_context, correct_post_context,
                 correct_current_response, swapped_context, swapped_current_response,
                 random_utt, swap_pre, rand_current):

        self.guid = guid
        self.utterance_id = utterance_id
        self.correct_pre_context = correct_pre_context
        self.correct_post_context = correct_post_context
        self.correct_current_response = correct_current_response
        self.swapped_context = swapped_context
        self.swapped_current_response = swapped_current_response
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
        self.current_utterance_id = current_utterance_id
