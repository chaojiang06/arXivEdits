# Standard library imports
from datetime import timedelta, datetime
import time
import random
import csv
import json
from os.path import expanduser
import argparse

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import numpy as np
from tqdm import tqdm
from pytorch_transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
)

assert torch.cuda.is_available(), print("CUDA is not available!")

SEED = 123456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

my_device = torch.device("cuda:0")

max_span_size = 1


def write_json(path, file):
    with open(path, "w") as handle:
        json.dump(file, handle, indent=4)


def read_json(path):
    with open(path, "r") as handle:
        b = json.load(handle)

    return b


def read_txt(path):
    with open(path) as f:
        data = f.readlines()

    data = [i.strip() for i in data]
    return data


def use_model_to_inference_in_practice(test_lsents, test_rsents, model):
    predicted_alignment = []

    for test_i in range(len(test_lsents)):
        output_type, output_score, predect_sequence = model(
            test_lsents[test_i], test_rsents[test_i], None
        )

        # print(test_lsents[test_i])
        # print(test_rsents[test_i])
        # print(predect_sequence)
        # print(golden_sequence[test_i])

        sub_prediected_alignment = []

        for i in range(len(predect_sequence)):
            if predect_sequence[i] != 0:
                small_pair = [
                    "simple_{}".format(i),
                    "complex_{}".format(predect_sequence[i] - 1),
                ]

                sub_prediected_alignment.append(small_pair)
        predicted_alignment.append(sub_prediected_alignment)

    return predicted_alignment


def convert_stateID_to_spanID(stateID, sent_length):  # 0 is NULL state
    stateID = stateID - 1
    if stateID < 0:
        return (-1, -1)
    else:
        for span_length in range(1, max_span_size + 1):
            lower_bound = (span_length - 1) * sent_length - int(
                (span_length - 1) * (span_length - 2) / 2
            )
            upper_bound = span_length * sent_length - int(
                span_length * (span_length - 1) / 2
            )
            if stateID >= lower_bound and stateID < upper_bound:
                return (
                    stateID - lower_bound,
                    span_length,
                )  # return (spanID, span_Length)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    output_mode,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    mask_padding_with_zero=True,
):
    """Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for ex_index, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(
                tokens_a, tokens_b, max_seq_length - special_tokens_count
            )
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] *
                           padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            segment_ids = segment_ids + \
                ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
            )
        )
    return features


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = (
            input_mask  # differentiate which part is input, which part is padding
        )
        self.segment_ids = segment_ids  # differentiate different sentences
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_tensor_from_sent_pair(sentA, sentB, model, tokenizer, mode="train"):
    # fake_example = [InputExample(guid=111, text_a=sentA, text_b=sentB, label=None)]
    model.eval()
    fake_example = []
    for i in range(len(sentA)):
        fake_example.append(
            InputExample(guid=i, text_a=sentA[i],
                         text_b=sentB[i], label="good")
        )

    fake_example_features = convert_examples_to_features(
        fake_example,
        ["good", "bad"],
        128,
        tokenizer,
        "classification",
        cls_token_at_end=bool("bert" in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if "bert" in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool("bert" in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool("bert" in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if "bert" in ["xlnet"] else 0,
    )

    all_input_ids = torch.tensor(
        [f.input_ids for f in fake_example_features], dtype=torch.long
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in fake_example_features], dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in fake_example_features], dtype=torch.long
    )
    all_label_ids = torch.tensor(
        [f.label_id for f in fake_example_features], dtype=torch.long
    )

    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )
    # # batch = dataset[0]
    # # batch = tuple(t.to(my_device) for t in batch)
    #
    # # model.eval()
    #
    # inputs = {'input_ids': torch.stack([i[0] for i in dataset]).to(my_device),
    #           'attention_mask': torch.stack([i[1] for i in dataset]).to(my_device),
    #           'token_type_ids': torch.stack([i[2] for i in dataset]).to(my_device),
    #           # XLM and RoBERTa don't use segment_ids
    #           'labels': torch.stack([i[3] for i in dataset]).to(my_device)}
    #
    # outputs = model(input_ids=inputs["input_ids"], \
    #                 attention_mask=inputs["attention_mask"], \
    #                 token_type_ids=inputs["token_type_ids"], \
    #                 labels=None, \
    #                 )
    #
    # # outputs = outputs.data()
    # outputs = outputs[1][-1][:, 0, :]
    # outputs = outputs.data
    output_tensor = []
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=8)
    for batch in eval_dataloader:
        batch = tuple(t.to(my_device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
            }
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                labels=None,
            )

            output_tensor.append(outputs[-1][-1][:, 0, :].data)

    output_tensor = torch.cat(output_tensor)

    return output_tensor


class NeuralSentenceAligner(nn.Module):
    def __init__(self, sent_pait_to_cls_dict, bert_for_sent_seq_model, tokenizer):
        super(NeuralSentenceAligner, self).__init__()

        self.bert_for_sent_seq_model = bert_for_sent_seq_model
        self.tokenizer = tokenizer
        max_span_size = 1
        self.max_span_size = 1

        self.mlp1 = nn.Sequential(
            nn.Linear(768, 768), nn.Tanh(), nn.Linear(768, 1))
        self.mlp2 = nn.Sequential(nn.Linear(6, 1))
        self.sent_pait_to_cls_dict = sent_pait_to_cls_dict

        self.transition_matrix_dict = {}
        for len_B in range(max_span_size, 200, 1):
            extended_length_B = self.max_span_size * len_B - int(
                self.max_span_size * (self.max_span_size - 1) / 2
            )
            transition_matrix = np.zeros(
                (extended_length_B + 1, extended_length_B + 1, 6), dtype=float
            )
            for j in range(extended_length_B + 1):  # 0 is NULL state
                for k in range(
                    extended_length_B + 1
                ):  # k is previous state, j is current state
                    if k == 0 and j == 0:
                        transition_matrix[j][k][1] = 1
                    elif k > 0 and j == 0:
                        transition_matrix[j][k][2] = 1
                    elif k == 0 and j > 0:
                        transition_matrix[j][k][3] = 1
                    # elif k<=len_B and j<=len_B:
                    # 	transition_matrix[j][k][0] = np.absolute(j - k - 1)
                    elif k > len_B and j <= len_B:
                        transition_matrix[j][k][4] = 1
                    elif k <= len_B and j > len_B:
                        transition_matrix[j][k][5] = 1
                    else:
                        transition_matrix[j][k][0] = self.distortionDistance(
                            k, j, len_B
                        )
            self.transition_matrix_dict[extended_length_B] = transition_matrix

    def viterbi_decoder(
        self, emission_matrix, transition_matrix, len_A, extended_length_B
    ):
        """
        :param emission_matrix:  extended_length_A * (extended_length_B + 1), word/phrase pair interaction matrix
        :param transition_matrix: (extended_length_B + 1) * (extended_length_B + 1), state transition matrix
        :param len_A: source sentence length
        :param len_B: target sentence length
        :return:
        """
        emission_matrix = emission_matrix.data.cpu().numpy()
        transition_matrix = transition_matrix.data.cpu().numpy()
        T1 = np.zeros((len_A, extended_length_B + 1), dtype=float)
        T2 = np.zeros((len_A, extended_length_B + 1), dtype=int)
        T3 = np.zeros((len_A, extended_length_B + 1), dtype=int)
        for j in range(extended_length_B + 1):
            # + transition_matrix[j][len_B+1]
            T1[0][j] = emission_matrix[0][j - 1]
            T2[0][j] = -1
            T3[0][j] = 1  # span size

        visited_states = set()
        for i in range(1, len_A):
            global_max_val = float("-inf")
            global_max_idx = -1
            for j in range(extended_length_B + 1):
                # if j in visited_states: # add constraint here
                # 	continue
                max_val = float("-inf")
                for span_size in range(
                    1, min(i + 1, self.max_span_size) + 1
                ):  # span_size can be {1,2,3,4}
                    for k in range(extended_length_B + 1):
                        if i - span_size >= 0:
                            cur_val = (
                                T1[i - span_size][k]
                                + transition_matrix[j][k]
                                + emission_matrix[
                                    i
                                    - (span_size - 1)
                                    + (span_size - 1) * len_A
                                    - int((span_size - 1) *
                                          (span_size - 2) / 2)
                                ][j - 1]
                            )
                        else:
                            cur_val = emission_matrix[
                                i
                                - (span_size - 1)
                                + (span_size - 1) * len_A
                                - int((span_size - 1) * (span_size - 2) / 2)
                            ][j - 1]
                        if cur_val > max_val:
                            T1[i][j] = cur_val
                            T2[i][j] = k
                            T3[i][j] = span_size
                            max_val = cur_val
                if max_val > global_max_val:
                    global_max_val = max_val
                    global_max_idx = j
        # visited_states.add(global_max_idx)
        optimal_sequence = []
        max_val = float("-inf")
        max_idx = -1
        for j in range(extended_length_B + 1):
            if T1[len_A - 1][j] > max_val:
                max_idx = j
                max_val = T1[len_A - 1][j]
        # optimal_sequence = [max_idx] + optimal_sequence
        # for i in range(len_A - 1, 0, -1):
        # 	optimal_sequence = [T2[i][max_idx]] + optimal_sequence
        # 	max_idx = T2[i][max_idx]
        i = len_A - 1
        while i >= 0:
            optimal_element = [max_idx] * T3[i][max_idx]
            optimal_sequence = optimal_element + optimal_sequence
            new_i = i - T3[i][max_idx]
            new_max_idx = T2[i][max_idx]
            i = new_i
            max_idx = new_max_idx

        return optimal_sequence

    def _score_sentence(
        self, output_both, transition_matrix, golden_sequence, len_A, len_B
    ):
        # golden_sequence is a list of states: [1, 2, 3, 33, 33, 33, 8, 9, 10, 11, 41, 15]
        # print(golden_sequence)
        score = 0
        # print(output_both.size())
        gold_list = []
        tmp = golden_sequence[0:1]
        for i, item in enumerate(golden_sequence):
            if i == 0:
                continue
            if item == tmp[-1]:
                if len(tmp) == max_span_size:
                    gold_list.append((i - len(tmp), tmp, tmp[-1]))
                    tmp = [item]
                else:
                    tmp.append(item)
            else:
                gold_list.append((i - len(tmp), tmp, tmp[-1]))
                tmp = [item]
        gold_list.append((len_A - len(tmp), tmp, tmp[-1]))
        # print(gold_list)
        for start_i, span, item in gold_list:
            span_size = len(span)
            score += output_both[
                start_i
                + (span_size - 1) * len_A
                - int((span_size - 1) * (span_size - 2) / 2)
            ][item - 1]
            if start_i - 1 >= 0:
                score += transition_matrix[item][golden_sequence[start_i - 1]]
        return score

    def argmax(self, vec):
        # return the argmax as a python int
        _, idx = torch.max(vec, 1)
        return idx.item()

    def log_sum_exp(self, vec):
        max_score = vec[0, self.argmax(vec)]
        max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def _forward_alg(self, output_both, transition_matrix, len_A):
        target_size = output_both.size(1)
        forward_var1 = torch.full((1, target_size), 0).to(my_device)
        forward_var2 = torch.full((1, target_size), 0).to(my_device)
        forward_var3 = torch.full((1, target_size), 0).to(my_device)
        forward_var4 = torch.full((1, target_size), 0).to(my_device)
        tmp_forward_var1 = torch.full((1, target_size), 1).to(my_device)
        tmp_forward_var2 = torch.full((1, target_size), 1).to(my_device)
        tmp_forward_var3 = torch.full((1, target_size), 1).to(my_device)
        tmp_forward_var4 = torch.full((1, target_size), 1).to(my_device)
        # forward_var1 = forward_var1.to(my_device)
        # forward_var2 = forward_var2.to(my_device)
        # forward_var3 = forward_var3.to(my_device)
        # tmp_forward_var1 = tmp_forward_var1.to(my_device)
        # tmp_forward_var2 = tmp_forward_var2.to(my_device)
        # tmp_forward_var3 = tmp_forward_var3.to(my_device)
        for i in range(len_A):
            for span_size in range(1, min(i + 1, self.max_span_size) + 1):
                alphas_t = []
                if span_size == 1:
                    feat = output_both[i]
                    for j in range(target_size):
                        emit_score = feat[j -
                                          1].view(1, -1).expand(1, target_size)
                        # print(emit_score.size())
                        # sys.exit()
                        trans_score = transition_matrix[j]  # [:-1]
                        if i <= 0:
                            next_tag_var = forward_var1 + emit_score
                        else:
                            next_tag_var = forward_var1 + trans_score + emit_score
                        # print(next_tag_var.size())
                        # print(self.log_sum_exp(next_tag_var))
                        # print(self.log_sum_exp(next_tag_var).view(1))
                        # sys.exit()
                        alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
                    tmp_forward_var1 = torch.cat(alphas_t).view(1, -1)
                elif span_size == 2 and i >= 1:
                    feat = output_both[i - 1 + len_A]
                    for j in range(target_size):
                        emit_score = feat[j -
                                          1].view(1, -1).expand(1, target_size)
                        trans_score = transition_matrix[j]
                        if i <= 1:
                            next_tag_var = forward_var2 + emit_score
                        else:
                            next_tag_var = forward_var2 + trans_score + emit_score
                        alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
                    tmp_forward_var2 = torch.cat(alphas_t).view(1, -1)
                elif span_size == 3 and i >= 2:
                    feat = output_both[i - 2 + 2 * len_A - 1]
                    for j in range(target_size):
                        emit_score = feat[j -
                                          1].view(1, -1).expand(1, target_size)
                        trans_score = transition_matrix[j]
                        if i <= 2:
                            next_tag_var = forward_var3 + emit_score
                        else:
                            next_tag_var = forward_var3 + trans_score + emit_score
                        alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
                    tmp_forward_var3 = torch.cat(alphas_t).view(1, -1)
                elif span_size == 4 and i >= 3:
                    feat = output_both[i - 3 + 3 * len_A - 3]
                    for j in range(target_size):
                        emit_score = feat[j -
                                          1].view(1, -1).expand(1, target_size)
                        trans_score = transition_matrix[j]
                        if i <= 3:
                            next_tag_var = forward_var4 + emit_score
                        else:
                            next_tag_var = forward_var4 + trans_score + emit_score
                        alphas_t.append(self.log_sum_exp(next_tag_var).view(1))
                    tmp_forward_var4 = torch.cat(alphas_t).view(1, -1)

            forward_var4 = forward_var3
            forward_var3 = forward_var2
            forward_var2 = forward_var1
            if i == 0:
                forward_var1 = tmp_forward_var1
            elif i == 1:
                max_score = torch.max(
                    torch.max(tmp_forward_var1), torch.max(tmp_forward_var2)
                )
                forward_var1 = max_score + torch.log(
                    torch.exp(tmp_forward_var1 - max_score)
                    + torch.exp(tmp_forward_var2 - max_score)
                )
            elif i >= 2:
                max_score = torch.max(
                    torch.max(tmp_forward_var1), torch.max(tmp_forward_var2)
                )
                max_score = torch.max(max_score, torch.max(tmp_forward_var3))
                forward_var1 = max_score + torch.log(
                    torch.exp(tmp_forward_var1 - max_score)
                    + torch.exp(tmp_forward_var2 - max_score)
                    + torch.exp(tmp_forward_var3 - max_score)
                )
        # elif i >= 3:
        # 	max_score = torch.max(torch.max(tmp_forward_var1), torch.max(tmp_forward_var2))
        # 	max_score = torch.max(max_score, torch.max(tmp_forward_var3))
        # 	max_score = torch.max(max_score, torch.max(tmp_forward_var4))
        # 	forward_var1 = max_score + torch.log(
        # 		torch.exp(tmp_forward_var1 - max_score) + torch.exp(tmp_forward_var2 - max_score) + torch.exp(
        # 			tmp_forward_var3 - max_score) + torch.exp( tmp_forward_var4 - max_score))

        alpha = self.log_sum_exp(forward_var1)
        return alpha

    def distortionDistance(self, state_i, state_j, sent_length):
        start_i, size_i = convert_stateID_to_spanID(state_i, sent_length)
        start_j, size_j = convert_stateID_to_spanID(state_j, sent_length)
        return np.absolute(start_j - (start_i + size_i - 1) - 1)

    def forward(self, raw_input_A, raw_input_B, golden_sequence):
        """
        :param raw_input_A: (source, source_dep_tag, source_dep_tree)
        :param raw_input_B: (target, target_dep_tag, target_dep_tree)
        :return:
        """
        # embd_A: # of chunks in A * embedding_dim

        output_type = None
        output_score = None

        syntac_loss = 0

        len_A = len(raw_input_A)
        len_B = len(raw_input_B)
        extended_length_A = len_A
        extended_length_B = len_B

        focusCube = torch.ones(len_A, len_B, 768)
        focusCube_A = torch.ones(len_A, len_B, 768)
        focusCube_B = torch.ones(len_A, len_B, 768)

        # sent_pair_cls_dict = load_pickle_file("")

        sent_A_list = []
        sent_B_list = []
        for iii in range(len_A):
            for jjj in range(len_B):
                sent_A_list.append(raw_input_A[iii])
                sent_B_list.append(raw_input_B[jjj])
                # aa = self.sent_pait_to_cls_dict[(raw_input_A[iii], raw_input_B[jjj])]
                # bb = get_tensor_from_sent_pair([raw_input_B[jjj]], [raw_input_A[iii]], \
                #                           self.bert_for_sent_seq_model, self.tokenizer)
                # if torch.allclose(aa, bb.cpu(), atol=1e-04) == False:
                #     print("aa != bb")

        tensor_matrix = get_tensor_from_sent_pair(
            sent_A_list, sent_B_list, self.bert_for_sent_seq_model, self.tokenizer
        )

        focusCube = tensor_matrix.view(len_A, len_B, -1)

        focusCube = F.pad(focusCube, (0, 0, 0, 1), "constant", 0)

        focusCube = focusCube.to(my_device)
        # print(focusCube.shape)
        output_both = self.mlp1(focusCube).squeeze(
            2
        )  # extended_length_A * (extended_length_B + 1)

        # output_both = self.mlp1(focusCube).squeeze()  # extended_length_A * (extended_length_B + 1)
        pair_loss = 0

        transition_matrix = Variable(
            torch.from_numpy(self.transition_matrix_dict[extended_length_B])
        ).type(torch.FloatTensor)
        transition_matrix = transition_matrix.to(my_device)
        # transition_matrix=self.mlp2(transition_matrix) * 0 + 1 # this is interesting

        transition_matrix = self.mlp2(transition_matrix)  # this is interesting

        transition_matrix = transition_matrix.view(
            transition_matrix.size(0), transition_matrix.size(1)
        )
        if self.training:
            forward_score = self._forward_alg(
                output_both, transition_matrix, len_A)
            gold_score = self._score_sentence(
                output_both, transition_matrix, golden_sequence, len_A, len_B
            )
            # print(forward_score, gold_score)
            return (
                forward_score - gold_score + syntac_loss + pair_loss * 0.1,
                output_type,
                output_score,
            )
        else:
            return_sequence = self.viterbi_decoder(
                output_both, transition_matrix, len_A, extended_length_B
            )
            return output_type, output_score, return_sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["inference"], help="set the running mode")
    parser.add_argument(
        "--corpus",
        choices=["arxiv"],
        default="arxiv",
        help="select the model that trained on the corresponding corpus",
    )
    parser.add_argument(
        "--direction",
        choices=[
            "bi-direction-union",
            "bi-direction-intersection",
        ],
        default="bi-direction-union",
        help="select which direction to align the corpus",
    )
    parser.add_argument(
        "--input",
        default="/nethome/cjiang95/share5/research_7_arxiv_alignment/20221114_1_github_repo/arXivEdits/code/pipeline/raw_input.json",
        help="the simple side of the corpus, where each line is a sentence",
    )
    parser.add_argument(
        "--output",
        default="/nethome/cjiang95/share5/research_7_arxiv_alignment/20221114_1_github_repo/arXivEdits/code/pipeline/step1.json",
        help="the path to output the alignment pairs"
    )
    args = parser.parse_args()

    # what should I do here?
    # step 1, load the data into two sentence lists

    data = read_json(args.input)
    lsents = []
    rsents = []
    for k, v in data.items():
        lsents.append(v["simple"])
        rsents.append(v["complex"])

    # print("start running")

    # step 2, load the model
    # Bert related
    MODEL_CLASSES = {
        "bert": (BertConfig, BertForSequenceClassification, BertTokenizer)}
    config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]

    device = torch.device("cuda")

    if args.corpus == "arxiv":
        # print("loading arxiv BERT model")
        tokenizer = tokenizer_class.from_pretrained(
            "../../checkpoints/arxiv-sentence-alignment",
            do_lower_case=True,
        )
        bert_for_sent_seq_model = model_class.from_pretrained(
            "../../checkpoints/arxiv-sentence-alignment",
            output_hidden_states=True,
        )

    bert_for_sent_seq_model.to(device)
    bert_for_sent_seq_model.eval()
    # Bert related end

    # step 3, do inference
    model = NeuralSentenceAligner(
        sent_pait_to_cls_dict="",
        bert_for_sent_seq_model=bert_for_sent_seq_model,
        tokenizer=tokenizer,
    )

    if args.corpus == 'arxiv':
        model.load_state_dict(
            torch.load(
                "/coc/pskynet5/cjiang95/research_7_arxiv_alignment/20220606_1_remake_alignment_table/crf_output/model_task2_epoch_2_state_dict_0910.pkl"
            )
        )

    model = model.to(my_device)
    model.eval()

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(parameters, lr=2e-5, eps=1e-8)

    # step 4, output

    predicted_alignment_1 = use_model_to_inference_in_practice(
        lsents, rsents, model)

    # print("predicted_alignment_1", predicted_alignment_1)

    if args.direction in ["bi-direction-union", "bi-direction-intersection"]:
        model = NeuralSentenceAligner(
            sent_pait_to_cls_dict="",
            bert_for_sent_seq_model=bert_for_sent_seq_model,
            tokenizer=tokenizer,
        )
        model.load_state_dict(
            torch.load(
                "/coc/pskynet5/cjiang95/research_7_arxiv_alignment/20220606_1_remake_alignment_table/crf_output_reversed/model_task2_epoch_2_state_dict_0910.pkl"
            )
        )

        model = model.to(my_device)
        model.eval()

        parameters = filter(lambda p: p.requires_grad, model.parameters())

        optimizer = torch.optim.Adam(parameters, lr=2e-5, eps=1e-8)

        # step 4, output

        predicted_alignment_2 = use_model_to_inference_in_practice(
            rsents, lsents, model)

        tmp = []
        for idx_i, i in enumerate(predicted_alignment_2):
            subtmp = []
            for idx_j, j in enumerate(i):
                subtmp.append(['simple_{}'.format(j[1].split("_")[-1]),
                              'complex_{}'.format(j[0].split("_")[-1])])
            tmp.append(subtmp)

        predicted_alignment_2 = tmp

        # print("predicted_alignment_2", predicted_alignment_2)

    predicted_alignment_final = []
    if args.direction == "bi-direction-union":
        for i, j in zip(predicted_alignment_1, predicted_alignment_2):
            tmp = []
            for ii in i + j:
                if ii not in tmp:
                    tmp.append(ii)

            predicted_alignment_final.append(tmp)
        # print("bi-direction-union, predicted_alignment_final",
            #   predicted_alignment_final)

    elif args.direction == 'bi-direction-intersection':
        for i, j in zip(predicted_alignment_1, predicted_alignment_2):
            tmp = []
            for ii in i:
                if ii in j:
                    tmp.append(ii)
            predicted_alignment_final.append(tmp)
        # print("bi-direction-intersection, predicted_alignment_final",
            #   predicted_alignment_final)

    output = {}
    sent_counter = 0
    article_counter = 0
    for k, v in data.items():
        simple = v["simple"]
        complex = v["complex"]
        for idx_j, j in enumerate(predicted_alignment_final[article_counter]):
            output[sent_counter] = {
                "easy-or-hard": "easy",
                "sentence-pair-index": sent_counter,
                "sentence-1": simple[int(j[0].split("_")[-1])],
                "sentence-2": complex[int(j[1].split("_")[-1])],
                "edits-combination-0": {},
                "edits-combination-1": {},
                "edits-combination-2": {},
                "arxiv-id": article_counter,
                "sentence-1-level": "simple",
                "sentence-2-level": "complex"
            }

            sent_counter += 1
        article_counter += 1
    # print("output", output)
    write_json(args.output, output)
