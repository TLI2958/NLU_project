"""
This version swaps the designation of source and target, as per paper. 
"""
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This module contains code adapted from `JointBERT`.
Copyright and license details can be found in `NOTICE.md`.
"""

# Modified from JointBERT:
# https://github.com/monologg/JointBERT/blob/master/model/modeling_jointbert.py

# Mainly From MASSIVE, slight modifications: 
# https://github.com/alexa/massive/blob/main/src/massive/models/xlmr_ic_sf.py
# https://github.com/alexa/massive/blob/main/src/massive/utils/training_utils.py
# https://github.com/alexa/massive/blob/main/src/massive/loaders/collator_ic_sf.py


import torch
from torch import nn
from transformers import (
    XLMRobertaModel,
    XLMRobertaConfig,
    XLMRobertaTokenizer
)
import datasets
from datasets import load_dataset

import os
import json
import numpy as np
import sklearn.metrics as sklm
from seqeval.metrics import f1_score
import pickle 
import sys

from math import sqrt
import warnings
import sklearn.exceptions
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)



random_seed = 1012

torch.seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# note: we are using the dataset: 
# https://github.com/alexa/massive?tab=readme-ov-file

# collator function for MASSIVE intent classification and slot filling
# modified for para_dataloaderclass CollatorMASSIVEIntentClassSlotFill_para:
class CollatorMASSIVEIntentClassSlotFill_para:
    """
    Data collator for the MASSIVE intent classification and slot tasks
    Based on: https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/data/data_collator.py#L212

    :param tokenizer: The tokenizer
    :type tokenizer: transformers.PreTrainedTokenizerFast
    :param padding: True or 'longest' pads to longest seq in batch, 'max_length' to the specified
                    max_length, and False or 'do_not_pad' to not pad (default)
    :type padding: bool, str, or transformers.file_utils.PaddingStrategy
    :param max_length: max length for truncation and/or padding (optional)
    :type max_length: int
    :param pad_to_multiple_of: set the padding such that sequence is multiple of this (optional)
    :type pad_to_multiple_of: int
    """

    def __init__(self, tokenizer, max_length, padding='max_length', pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of

        self.col_chk = 0

    def __call__(self, batch):
        # On-the-fly tokenization and alignment -- do NOT use a pre-tokenized dataset

        tokenized_source = self.tokenizer(
            [item['utt'] for item in batch],
            truncation=True,
            is_split_into_words=True
        )

        tokenized_target = self.tokenizer(
            [item['target_utt'] for item in batch],
            truncation=True,
            is_split_into_words=True
        )
        
        # Align the labels with the tokenized utterance
        # adapted from here: https://huggingface.co/docs/transformers/custom_datasets#tok_ner
        for i, entry in enumerate(batch):
            label = entry['slots_num']
            word_ids = tokenized_source.word_ids(batch_index=i)  # source
            previous_word_idx = None
            source_ids = []
            # Set the special tokens to -100.
            for word_idx in word_ids:
                if word_idx is None:
                    source_ids.append(-100)
                # Only label the first token of a given word.
                elif word_idx != previous_word_idx:
                    source_ids.append(label[word_idx])
                    previous_word_idx = word_idx
                else:
                    source_ids.append(-100)
            
            
            # only use source 
            if 'slots_label' in tokenized_source:
                tokenized_source['slots_label'].append(source_ids)
            else:
                tokenized_source['slots_label'] = [source_ids]

        # Pad the inputs
        pad_tok_inputs = self.tokenizer.pad(
            tokenized_source,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        pad_tok_inputs_target = self.tokenizer.pad(
            tokenized_target,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        # Pad the slot labels
        pad_tok_inputs["source"] = pad_tok_inputs["input_ids"]
        pad_tok_inputs["target"] = pad_tok_inputs_target["input_ids"]
        del pad_tok_inputs["input_ids"]

        sequence_length = torch.tensor(pad_tok_inputs["source"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            pad_tok_inputs['slots_label'] = [
                list(label) + [-100] * (sequence_length - len(label)) \
                               for label in pad_tok_inputs['slots_label']
            ]
        else:
            pad_tok_inputs['slots_label'] = [
                [-100] * (sequence_length - len(label)) + list(label) \
                 for label in pad_tok_inputs['slots_label']
            ]

        # Add in the intent labels
        pad_tok_inputs["intent_label"] = [item['intent_num'] for item in batch]
        
        order_of_key = ["target", "source", "slots_label", "intent_label"] # swap target & source to see the difference
        # order_of_key = ["source", "slots_label", "intent_label", "attention_mask"]

        # Convert to PyTorch tensors
        res = {k: torch.tensor(pad_tok_inputs[k], dtype=torch.int64) for k in order_of_key} 
        res.update({'attention_mask': torch.tensor(pad_tok_inputs_target['attention_mask'], dtype=torch.int64)})
        return res 
        
class CollatorMASSIVEIntentClassSlotFill:
    """
    Data collator for the MASSIVE intent classification and slot tasks
    Based on: https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/data/data_collator.py#L212

    :param tokenizer: The tokenizer
    :type tokenizer: transformers.PreTrainedTokenizerFast
    :param padding: True or 'longest' pads to longest seq in batch, 'max_length' to the specified
                    max_length, and False or 'do_not_pad' to not pad (default)
    :type padding: bool, str, or transformers.file_utils.PaddingStrategy
    :param max_length: max length for truncation and/or padding (optional)
    :type max_length: int
    :param pad_to_multiple_of: set the padding such that sequence is multiple of this (optional)
    :type pad_to_multiple_of: int
    """

    def __init__(self, tokenizer, max_length, padding='max_length', pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of

        self.col_chk = 0

    def __call__(self, batch):
        # On-the-fly tokenization and alignment -- do NOT use a pre-tokenized dataset

        tokenized_inputs = self.tokenizer(
            [item['utt'] for item in batch],
            truncation=True,
            is_split_into_words=True
        )
        
        # Align the labels with the tokenized utterance
        # adapted from here: https://huggingface.co/docs/transformers/custom_datasets#tok_ner
        for i, entry in enumerate(batch):
            label = entry['slots_num']
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to respective word
            previous_word_idx = None
            label_ids = []
            # Set the special tokens to -100.
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                # Only label the first token of a given word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                    previous_word_idx = word_idx
                else:
                    label_ids.append(-100)


            if 'slots_num' in tokenized_inputs:
                tokenized_inputs['slots_num'].append(label_ids)
            else:
                tokenized_inputs['slots_num'] = [label_ids]

        # Pad the inputs
        pad_tok_inputs = self.tokenizer.pad(
            tokenized_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        # Pad the slot labels
        sequence_length = torch.tensor(pad_tok_inputs["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            pad_tok_inputs['slots_num'] = [
                list(label) + [-100] * (sequence_length - len(label)) \
                               for label in pad_tok_inputs['slots_num']
            ]
        else:
            pad_tok_inputs['slots_num'] = [
                [-100] * (sequence_length - len(label)) + list(label) \
                 for label in pad_tok_inputs['slots_num']
            ]

        # Add in the intent labels
        pad_tok_inputs["intent_num"] = [item['intent_num'] for item in batch]

        # Convert to PyTorch tensors
        order_of_key = ["input_ids", "slots_num", "intent_num", "attention_mask"]
        # Convert to PyTorch tensors
        return {k: torch.tensor(pad_tok_inputs[k], dtype=torch.int64) for k in order_of_key}

# compute metrics
def create_compute_metrics(intent_labels = None, slot_labels = None, ignore_labels=None, 
                           metrics='all', average = 'micro'):
    """
    Create a `compute_metrics` function for this task

    :param intent_labels: A dictionary mapping each intent's numerical index to the intent
    :type slot_labels: dict
    :param slot_labels: A dictionary mapping each slot's numerical index to the slot
    :type slot_labels: dict
    :param conf: The MASSIVE configuration object
    :type conf: massive.Configuration
    :param ignore_labels: The labels to ignore
    :type ignore_labels: list or str
    :param metrics: The metrics to calculate
    :type metrics: list or str
    :return: the `compute_metrics` function
    :rtype: Callable
    """

    # Determine any labels that should be ignored when calculating F1 score (EX: Other)
    ignore_labels = [] if ignore_labels is None else ignore_labels
    ignore_num_lab = [int(k) for k, v in slot_labels.items() if v in ignore_labels]

    if type(metrics) != list:
        metrics = [metrics]
    def compute_metrics(p):
        
        intent_preds = p.predictions[0]
        slot_preds = p.predictions[1]

        intent_label_tuple = p.label_ids[0]
        slot_label_tuple = p.label_ids[1]
        intent_preds_am = [torch.argmax(x, axis = -1) for x in intent_preds]
        slot_preds_am = [torch.argmax(x, axis=-1) for x in slot_preds]
        attn_masks = p.attn_masks
        # merge -100, which we used for the subsequent subwords in a full word after tokenizing
        labels_merge = [-100]

        return eval_preds(
                pred_intents=intent_preds_am,
                lab_intents=intent_label_tuple,
                pred_slots=slot_preds_am,
                lab_slots=slot_label_tuple,
                eval_metrics=metrics,
                labels_merge=labels_merge,
                labels_ignore=ignore_num_lab,
                pad='Other',
                attn_masks = attn_masks,
                average = average,
            )
    return compute_metrics

# eval function: 
def eval_preds(pred_intents=None, lab_intents=None, pred_slots=None, lab_slots=None,
               eval_metrics='all', labels_ignore='Other', labels_merge=None, pad='Other', 
              attn_masks = None, average = 'micro'):
    """
    Function to evaluate the predictions from a model

    :param pred_intents: a list of predicted intents
    :type pred_intents: list
    :param lab_intents: a list of intents labels (ground truth)
    :type lab_intents: list
    :param pred_slots: a list of predicted slots, where each entry is a list of token-based slots
    :type pred_slots: list
    :param lab_slots: a list of slots labels (ground truth)
    :type lab_slots: list
    :param eval_metrics: The metrics to include. Options are 'all', 'intent_acc', 'ex_match_acc',
                         'slot_micro_f1'
    :type eval_metrics: str
    :param labels_ignore: The labels to ignore (prune away). Default: ['Other']
    :type labels_ignore: str or list
    :param labels_merge: The labels to merge leftward (i.e. for tokenized inputs)
    :type labels_merge: str or list
    :param pad: The value to use when padding slot predictions to match the length of ground truth
    :type pad: str
    """

    results = {}

    # Check lengths
    if pred_intents is not None and lab_intents is not None:
        assert len(pred_intents) == len(lab_intents),"pred_intents and lab_intents must be same len"
    if pred_slots is not None and lab_slots is not None:
        assert len(pred_slots) == len(lab_slots), "pred_slots and lab_slots must be same length"
    
    
    if ('intent_acc' in eval_metrics) or ('all' in eval_metrics):
        intent_acc = sklm.accuracy_score(lab_intents, pred_intents)
        results['intent_acc'] = intent_acc
        # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
        results['intent_acc_stderr'] = sqrt(intent_acc*(1-intent_acc)/len(pred_intents))

    if lab_slots is not None and pred_slots is not None:
        bio_slot_labels, bio_slot_preds = [], []
        # j = 0
        for lab, pred, attn in zip(lab_slots, pred_slots, attn_masks):
            pred = list(pred)
            attn = list(attn)
            # Pad or truncate prediction as needed using `pad` arg
            if type(pred) == list:
                pred = pred[:len(lab)] + [pad]*(len(lab) - len(pred))
            # Fix for Issue 21 -- subwords after the first one from a word should be ignored
            for i, x in enumerate(lab):
                if x.item() == -100:
                    pred[i] = torch.tensor(-100)

            # convert to BIO
            bio_slot_labels.append(
                convert_to_bio(lab, outside=labels_ignore, labels_merge=labels_merge, attn_mask = attn)
            )
            # print('bio_lab:', bio_slot_labels[j][:10])
            bio_slot_preds.append(
                convert_to_bio(pred, outside=labels_ignore, labels_merge=labels_merge, attn_mask = attn)
            )
            # print('bio_preds:', bio_slot_preds[j][:10])
            # j += 1
        # raise ValueError()
        # with open('bio_slot_labels.pkl', 'wb') as f:
        #     pickle.dump(bio_slot_labels, f)

        # with open('bio_slot_preds.pkl', 'wb') as f:
        #     pickle.dump(bio_slot_preds, f)
            
    print('finish conversion to bio...')

    if ('slot_micro_f1' in eval_metrics) or ('all' in eval_metrics):

        # from seqeval
        smf1 = f1_score(bio_slot_labels, bio_slot_preds, average = average)
        results['slot_micro_f1'] = smf1
        # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
        total_slots = sum([len(x) for x in bio_slot_preds])
        results['slot_micro_f1_stderr'] = sqrt(smf1*(1-smf1)/total_slots)

    if ('ex_match_acc' in eval_metrics) or ('all' in eval_metrics):
        # calculate exact match accuracy (~0.01 seconds)
        matches = 0
        denom = 0
        for p_int, p_slot, l_int, l_slot in zip(pred_intents,
                                                bio_slot_preds,
                                                lab_intents,
                                                bio_slot_labels):

            if (p_int == l_int) and (p_slot == l_slot):
                matches += 1
            denom += 1
        emacc = matches / denom

        results['ex_match_acc'] = emacc
        # Assuming normal distribution. Multiply by z (from "z table") to get confidence int
        results['ex_match_acc_stderr'] = sqrt(emacc*(1-emacc)/len(pred_intents))

    return results

def convert_to_bio(seq_tags, outside='Other', labels_merge=None, attn_mask = None):
    """
    Converts a sequence of tags into BIO format. EX:

        ['city', 'city', 'Other', 'country', -100, 'Other']
        to
        ['B-city', 'I-city', 'O', 'B-country', 'I-country', 'O']
        where outside = 'Other' and labels_merge = [-100]

    :param seq_tags: the sequence of tags that should be converted
    :type seq_tags: list
    :param outside: The label(s) to put outside (ignore). Default: 'Other'
    :type outside: str or list
    :param labels_merge: The labels to merge leftward (i.e. for tokenized inputs)
    :type labels_merge: str or list
    :return: a BIO-tagged sequence
    :rtype: list
    """
    if attn_mask is None:
        raise ValueError("Attention mask cannot be None!")
    seq_tags = [str(x.item()) for x in seq_tags]
    outside = [outside] if isinstance(outside, str) else outside
    outside = [str(x.item()) for x in outside]

    if labels_merge:
        labels_merge = [labels_merge] if type(labels_merge) != list else labels_merge
        labels_merge = [str(x) for x in labels_merge]
    else:
        labels_merge = []
    
    bio_tagged = []
    prev_tag = None
    for i, tag in enumerate(seq_tags):
        if attn_mask[i] == 1:
            if tag in outside:
                bio_tagged.append('O')
                prev_tag = None
            elif tag != prev_tag and tag not in labels_merge:
                bio_tagged.append('B-' + tag)
                prev_tag = tag
            elif tag == prev_tag or tag in labels_merge:
                if prev_tag in outside or prev_tag is None:
                    bio_tagged.append('O')
                else:
                    bio_tagged.append('I-' + prev_tag)
        else:
            bio_tagged.append('O')

    return bio_tagged

# TODO: convert to text
def output_predictions(outputs, intent_labels, slot_labels, tokenizer=None,
                       combine_slots=True, remove_slots=None, add_pred_parse=True,
                       save_to_file=True, file_dir = None):
    """
    :param outputs: The outputs from the model
    :type outputs: named_tuple
    :param intent_labels: A dictionary mapping each intent's numerical index to the intent
    :type slot_labels: dict
    :param slot_labels: A dictionary mapping each slot's numerical index to the slot
    :type slot_labels: dict
    :param conf: The MASSIVE configuration object
    :type conf: massive.Configuration
    :param tokenizer: The tokenizer
    :type tokenizer: PreTrainedTokenizerFast
    :param combine_slots: Whether or not to combine adjacent same-slotted tokens to one slot
    :type combine_slots: bool
    :param remove_slots: Slots to remove. Default ['Other']
    :type remove_slots: list
    :param add_pred_parse: Whether to add the SLURP-style parsed output
    :type add_pred_parse: bool
    :param save_to_file: Whether to save predictions to the file given in the config
    :type save_to_file: bool
    """

    remove_slots = ['Other'] if not remove_slots else remove_slots

    with open(file_dir + '.pkl') as f:
        outputs = pickle.load(f)

    final_outputs = []

    # Create strings of the slot predictions
    intent_preds, slot_preds = outputs.predictions[0], outputs.predictions[1]
    intent_preds_am = [np.argmax(x) for x in intent_preds]
    intent_preds_str = [intent_labels[str(x)] for x in intent_preds_am]

    slot_preds_am = [np.argmax(x, axis=1) for x in slot_preds]
    slot_preds_str = []
    for example in slot_preds_am:
        slot_preds_str.append([slot_labels[str(x)] for x in example])

    # Iterate through the examples
    for eyed, loc, utt, tok_utt, intent_pred, slot_pred, subword_align in zip(
        outputs.ids,
        outputs.locales,
        outputs.utts,
        outputs.tok_utts,
        intent_preds_str,
        slot_preds_str,
        outputs.subword_aligns):

        line = {}
        line['id'], line['locale'], line['utt'], line['pred_intent'] = eyed,loc,utt,intent_pred

        # Determine slot predictions
        running_detok_idx, tok, slot, slots = -1, '', '', []
        for tok_idx, detok_idx in enumerate(subword_align):
            if detok_idx is None:
                continue
            # Combine the subwords that had been broken up
            if detok_idx == running_detok_idx:
                # If there is a \u2581 within what was a single "word" from the input data,
                # then it's probably from a zero-width space, \u200b, which the tokenizer
                # converted to a space. We don't want these extra spaces, so they are removed
                if replace_zwsp:
                    tok_repl = tok_utt[tok_idx].replace(u'\u2581',u'\u200b')
                else:
                    tok_repl = tok_utt[tok_idx]
                tok += tok_repl

            # Record the token and slot and start a new one
            else:
                if running_detok_idx != -1:
                    tok = tok.replace('▁',' ')
                    tok = tok.strip()
                    slots.append((tok, slot))
                slot = slot_pred[tok_idx]
                tok = tok_utt[tok_idx]
            running_detok_idx = detok_idx
        # Add the last token and slot
        tok = tok.replace('▁',' ')
        tok = tok.strip()
        slots.append((tok, slot))

        line['pred_slots'] = slots
        final_outputs.append(line)



    for line in final_outputs:
        slots = []
        for tup in line['pred_slots']:
            if not slots:
                slots.append(tup)
                slots_idx = 0
            # if slot the same as previous token, combine
            elif tup[1] == slots[slots_idx][1]:
                slots[slots_idx] = (slots[slots_idx][0] + ' ' + tup[0], slots[slots_idx][1])
            # otherwise add to end
            else:
                slots.append(tup)
                slots_idx += 1

        # Create a SLURP-like version of each utterance
        if add_pred_parse:
            parse = ''
            for slot in slots:
                if slot[1] in remove_slots:
                    parse += ' ' + slot[0]
                else:
                    parse += ' [' + slot[1] + ' : ' + slot[0] + ']'
            line['pred_annot_utt'] = parse.strip()

        # If adjacent tokens have the same slot, combine them
        if combine_slots:
            line['pred_slots'] = slots

        # Remove slots in the remove_slots list
        if remove_slots:
            line['pred_slots'] = [x for x in line['pred_slots'] if x[1] not in remove_slots]


    # True to output escaped unicode codes or False to output unicode
    ensure_ascii = conf.get('test.predictions_ensure_ascii', default=False)

    if save_to_file:
        with open(conf.get('test.predictions_file'), 'w', encoding='utf-8') as f:
            for line in final_outputs:
                f.write(json.dumps(line, ensure_ascii=ensure_ascii) + '\n')

    return final_outputs


def convert_eval(intent_labels, slot_labels, lang = 'zh', src = 'en'):
    with open('/kaggle/input' + f'/{src}-train/{src}.intents', 'r', encoding = 'UTF-8') as file:
        intent_labels_map = json.load(file)
    
    with open('/kaggle/input'+ f'/{src}-train/{src}.slots', 'r', encoding = 'UTF-8') as file:
        slot_labels_map = json.load(file)
    
    with open('/kaggle/input' + f'/{lang}-train/{lang}.intents', 'r', encoding = 'UTF-8') as file:
        zh_intent_labels_map = json.load(file)
    
    with open('/kaggle/input' + f'/{lang}-train/{lang}.slots', 'r', encoding = 'UTF-8') as file:
        zh_slot_labels_map =json.load(file)
    
    label_to_pred_idx = {v: k for k, v in intent_labels_map.items()}
    conversion_intent_map = {idx: int(label_to_pred_idx[label]) for idx, label in zh_intent_labels_map.items()}
    
    label_to_pred_idx = {v: k for k, v in slot_labels_map.items()}
    conversion_slot_map = {idx: int(label_to_pred_idx[label]) for idx, label in zh_slot_labels_map.items()}
    
    converted_intent_labels = [torch.tensor(conversion_intent_map.get(str(p.item()), -100)) for p in intent_labels]  
    converted_slot_labels = [torch.tensor([conversion_slot_map.get(str(s.item()), -100) for s in slot_seq]) for slot_seq in slot_labels]
    
    slot_tensor = torch.stack(converted_slot_labels) 
    intent_tensor = torch.stack(converted_intent_labels)
    return intent_tensor.to(device), slot_tensor.to(device)
    

def convert_train(example, src = 'en'):
    with open('/kaggle/input' + f'/{src}-train/{src}.intents', 'r', encoding = 'UTF-8') as file:
        intent_labels_map = json.load(file)
    
    with open('/kaggle/input'+ f'/{src}-train/{src}.slots', 'r', encoding = 'UTF-8') as file:
        slot_labels_map = json.load(file)
    
        
    label_to_pred_intent_idx = {v: k for k, v in intent_labels_map.items()}
    
    label_to_pred_slot_idx = {v: k for k, v in slot_labels_map.items()}
    
    converted_intent_labels = [int(label_to_pred_intent_idx.get(p, -100)) for p in example['target_intents']]  
    converted_slot_labels = [[int(label_to_pred_slot_idx.get(s, -100)) for s in slot_seq] for slot_seq in example['target_slots']]
    example['target_intents'], example['target_slots'] = converted_intent_labels, converted_slot_labels
    return example
       

# heavily modified from soft-align implementation in multiatis: 
# https://github.com/amazon-science/multiatis/blob/main/code/scripts/bert_soft_align.py
# implementation of soft-align in paper: https://arxiv.org/pdf/2004.14353
# original code is in mxnet, this is a port to pytorch + transformers

# also from MASSIVE utils:
# https://github.com/alexa/massive/blob/main/src/massive/utils/


from torch.cuda.amp import autocast, GradScaler

# largely modified from soft-align implementation in multiatis: 
# https://github.com/amazon-science/multiatis/blob/main/code/scripts/bert_soft_align.py
# implementation of soft-align in paper: https://arxiv.org/pdf/2004.14353
# original code is in mxnet, this is a port to pytorch + transformers


from collections import namedtuple
import os
from tqdm import tqdm
import numpy as np
import random
import argparse
import warnings
from copy import deepcopy

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


from transformers import AutoTokenizer, XLMRobertaModel
from datasets import load_dataset, Dataset


class MultiTaskICSL(nn.Module):
    """Model for IC/SL task.

    The model feeds token ids into XLM-RoBERTa to get the sequence
    representations, then apply two dense layers for IC/SL task.
    """

    def __init__(self, base_model, vocab_size, num_slot_labels, num_intents, hidden_size=768, 
                 dropout=.1,):
        super(MultiTaskICSL, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        # IC/SL classifier
        self.slot_classifier = nn.Linear(hidden_size,
                                         num_slot_labels,)
        self.intent_classifier = nn.Linear(hidden_size,
                                           num_intents,)
        # LM output layer: reconstruction of the source
        self.embedding_weight = self.base_model.get_input_embeddings().weight

        self.lm_output_layer = nn.Linear(hidden_size,
                                         vocab_size,    
                                         bias=False)
        self.lm_output_layer.weight = self.embedding_weight # weight tying

        self.attention_layer = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1,
                                                                       dropout = 0.02)
        self.layerNorm = nn.LayerNorm(hidden_size, eps=1e-6)
        # after dropout & layer norm
        self.ffn_layer = nn.Sequential(
                                       nn.Linear(hidden_size, hidden_size*2),
                                       nn.GELU(),
                                       nn.Linear(hidden_size * 2, hidden_size),)


    def encode(self, inputs, attn_mask = None):
        # XLM-RoBERTa: 0 if masked, 1 not, according to 
        # https://huggingface.co/docs/transformers/en/model_doc/xlm-roberta#transformers.XLMRobertaModel.forward.attention_mask
        encoded = self.base_model(inputs, attention_mask = attn_mask)
        encoded = self.dropout(encoded.last_hidden_state)
        return encoded

    def forward(self, inputs, attn_mask = None): 
        """Generate logits given input sequences.
        
        Parameters
        ----------
        inputs : (batch_size, seq_length)
            Input words for the sequences.
        attn_mask : (batch_size, seq_length)
            To mask the padded tokens.

        Returns
        -------
        intent_prediction: (batch_size, num_intents)
        slot_prediction : (batch_size, seq_length, num_slot_labels)
        """
        if attn_mask is not None:
            hidden = self.encode(inputs = inputs, attn_mask = attn_mask)
        
        intent_prediction = self.intent_classifier(hidden[:, 0, :])
        slot_prediction = self.slot_classifier(hidden[:, 1:, :])
        return intent_prediction, slot_prediction

    def translate_and_predict(self, source, target, source_attn_mask = None):
        """Generate logits given input sequences.

        Parameters
        ----------
        source : (batch_size, src_seq_length)
            Input words for the source sequences.
        target : (batch_size, tgt_seq_length)
            Input words for the target sequences.
        attn_mask: (batch_size, src_seq_length)  

        Returns
        -------
        translation : 
            Shape (batch_size, tgt_seq_length, vocab_size)
        intent_prediction: 
            Shape (batch_size, num_intents)
        slot_prediction :
            Shape (batch_size, tgt_seq_length, num_slot_labels)
        """
        src_encoded = self.encode(source, source_attn_mask)
        tgt_embed = self.base_model.embeddings(target)
        attention_mask = source_attn_mask.to(torch.float32)
        attention_mask = (1 - attention_mask) * (-1e9)
        # from https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
        attn_output, _ = self.attention_layer(query = tgt_embed, key = src_encoded, value = src_encoded, 
                                             key_padding_mask = attention_mask.transpose(0,1))
        
        # mxnet ref: https://nlp.gluon.ai/_modules/gluonnlp/model/attention_cell.html#AttentionCell
        # attn_output_ = self.dropout(attn_output)
        attn_output_ = self.layerNorm(attn_output)
        decoded = self.ffn_layer(attn_output_)
        
        decoded = self.dropout(decoded)
        decoded += attn_output_
        decoded = self.layerNorm(decoded)
        
        translation = self.lm_output_layer(decoded[:, 1:, :])

        # IC and SL
        intent_prediction = self.intent_classifier(src_encoded[:, 0, :])
        slot_prediction = self.slot_classifier(attn_output[:, 1:, :])
        return translation, intent_prediction, slot_prediction


def seed_everything(seed=1012):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# checkpoint
def save_checkpoint(model, optimizer, epoch, args, loader_name = None):
    if loader_name is None:
        checkpoint_path = f'{args.save_dir}/trained_{args.label}_checkpoint_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'lr_scheduler_state_dict': lr_scheduler.state.dict(),
        }, checkpoint_path)
    else:
        checkpoint_path = f'{args.save_dir}/trained_{args.label}_{loader_name}_checkpoint_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'lr_scheduler_state_dict': lr_scheduler.state.dict(),
        }, checkpoint_path)

    
def load_checkpoint(model, optimizer, args, loader_name = 'labeled'):
    if loader_name is None:
        checkpoint_path = f'{args.save_dir}/trained_{args.label}_{loader_name}_checkpoint_{args.epoch}.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location = torch.device(device))
            model.load_state_dict(checkpoint['model_state_dict'], )
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'], )
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler.state.dict'],)
            epoch = checkpoint['epoch']
            print(f"Checkpoint found. Resuming training from epoch {epoch}.")
            return model, optimizer, epoch
        else:
            return model, optimizer, 0
    else:
        checkpoint_path = f'{args.save_dir}/trained_{args.label}_{loader_name}_checkpoint_{args.epoch}.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location = torch.device(device))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler.state.dict'],)
            epoch = checkpoint['epoch']
            print(f"Checkpoint found. Resuming training from epoch {epoch}.")
            return model, optimizer, epoch
        else:
            return model, optimizer, 0
            

def train(model, optimizer, lr_scheduler, train_dataloader, para_dataloader):
    # num_slot_labels & num_intents: according to https://arxiv.org/pdf/2204.08582
    # note 56 num_slot_labels! not 55!
    
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, 
                                                                  args, loader_name = 'labeled')

    model.to(device)
    ic_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    sl_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    mt_loss_fn = nn.CrossEntropyLoss(reduction='mean')

    paral_size = len(para_dataloader)
    label_size = len(train_dataloader)

    pbar = tqdm(range(max(1, start_epoch + 1), args.num_epochs + 1))
    scaler = GradScaler()
    for epoch in pbar:
        mt_loss, icsl_loss, step_loss = 0, 0, 0
        
        intent_preds = []
        slot_preds = []
        intent_labels = []
        slot_labels = []
        
        # train on parallel data
        model.train()
        for para_batch in tqdm(para_dataloader, 
                                  total=len(para_dataloader)):
            # swap source & target to match the paper
            source, target, slot_label, intent_label, source_attn_mask = para_batch.values()
            source, target, slot_label, intent_label, source_attn_mask = (source.to(device), 
                                                                         target.to(device), slot_label.to(device), 
                                                                         intent_label.to(device), 
                                                                         source_attn_mask.to(device))
            translation, intent_pred, slot_pred = model.translate_and_predict(source, 
                                                                              target, 
                                                                              source_attn_mask = source_attn_mask)
            
            intent_preds.append(intent_pred.detach().to('cpu'))
            slot_preds.append(slot_pred.detach().to('cpu'))
            intent_labels.append(intent_label.detach().to('cpu'))
            slot_labels.append(slot_label.detach().to('cpu'))

            ic_loss = ic_loss_fn(intent_pred, intent_label)
            # slot_loss = sl_loss_fn(slot_pred.view(-1, slot_pred.size(-1)), slot_label.view(-1)) 
            sl_loss = sl_loss_fn(slot_pred.transpose(1,2), slot_label[:, 1:])
            mce_loss = mt_loss_fn(translation.transpose(1,2), source[:, 1:]) # since zh-en
            loss = ic_loss + sl_loss + mce_loss
            icsl_loss += ic_loss.detach().to('cpu').item() + sl_loss.detach().to('cpu').item()
            mt_loss += mce_loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 

        save_checkpoint(model, optimizer, epoch, args, loader_name = 'parallel')
        print('saved checkpoint...')
        predictions = (torch.cat(intent_preds), torch.cat(slot_preds))
        label_ids = (torch.cat(intent_labels), torch.cat(slot_labels))

        eval_ = {'predictions': (torch.cat(intent_preds), torch.cat(slot_preds)),
                     'label_ids': (torch.cat(intent_labels), torch.cat(slot_labels))}
            
        pbar.set_postfix({'dataset': 'parallel',
                'train_loss': (icsl_loss + mt_loss) / paral_size , 
                  'icsl_loss': icsl_loss / paral_size,
                  'mt_loss': mt_loss / paral_size,})

        
        # train on labeled data
        for batch in tqdm(train_dataloader, 
                             total=len(train_dataloader)):
            inputs, slot_label, intent_label, attn_mask = batch.values()
            inputs, slot_label, intent_label, attn_mask = (inputs.to(device), slot_label.to(device), 
                                                           intent_label.to(device), attn_mask.to(device))
            
            intent_pred, slot_pred = model(inputs, attn_mask)
            ic_loss = ic_loss_fn(intent_pred, intent_label)
            # slot_loss = sl_loss_fn(slot_pred.view(-1, slot_pred.size(-1)), slot_label.view(-1))
            sl_loss = sl_loss_fn(slot_pred.transpose(1,2), slot_label[:,1:])
            loss = ic_loss + sl_loss
            step_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            intent_preds.append(intent_pred.detach().to('cpu'))
            slot_preds.append(slot_pred.detach().to('cpu'))
            intent_labels.append(intent_label.detach().to('cpu'))
            slot_labels.append(slot_label.detach().to('cpu'))

        
        save_checkpoint(model, optimizer, epoch, args, loader_name = 'labeled')        
        print('saved checkpoint...')
        predictions = (torch.cat(intent_preds), torch.cat(slot_preds))
        label_ids = (torch.cat(intent_labels), torch.cat(slot_labels))
        
        pbar.set_postfix({'dataset': 'labeled',
                          'train_loss': step_loss / (label_size)})
                          # 'intent_acc': res['intent_acc'],
                          # 'slot_f1': res['slot_micro_f1'],
                          # 'ex_match_acc': res['ex_match_acc']})
                
        with open(os.path.join(args.save_dir, 'train.log.pkl'), 'a') as f:
            f.write(f'\nepoch: {epoch}\tstep_loss: {step_loss / label_size}\t icls_loss: {icsl_loss / paral_size}\t mt_loss: {mt_loss / paral_size}\n')
    
        # if epoch%3 == 0:
        evaluate(model, eval_dataloader = train_eval_dataloader, train_eval= True)
        # lr_scheduler.step()


def evaluate(model, eval_dataloader, train_eval = False):
    """Evaluate the model on validation dataset.
        
    Should be held-out Chinese utterances
    using  `eval_preds` from massive_utils
    """
    ic_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    sl_loss_fn = nn.CrossEntropyLoss(reduction='mean')

    intent_preds = []
    slot_preds = []
    intent_labels = []
    slot_labels = []
    attn_masks = []

    eval_size = len(eval_dataloader)    

    model.to(device)
    model.eval()
    with torch.no_grad():
        step_loss = 0
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            inputs, slot_label, intent_label, attn_mask = map(lambda x: x.to(device), batch.values())
            # note zh & en have different mapping!
            intent_pred, slot_pred = model(inputs, attn_mask)
            intent_label, slot_label = convert_eval(intent_label, slot_label, lang = args.lang, src = args.src) 

            ic_loss = ic_loss_fn(intent_pred, intent_label)
            sl_loss = sl_loss_fn(slot_pred.transpose(1,2), slot_label[:,1:])
            loss = ic_loss + sl_loss
            step_loss += loss.item()
            
            intent_preds.append(intent_pred.detach().to('cpu'))
            slot_preds.append(slot_pred.detach().to('cpu'))
            intent_labels.append(intent_label.detach().to('cpu'))
            slot_labels.append(slot_label.detach().to('cpu'))
            attn_masks.append(torch.tensor(attn_mask.detach().to('cpu').tolist()))


        predictions = (torch.cat(intent_preds), torch.cat(slot_preds))
        label_ids = (torch.cat(intent_labels), torch.cat(slot_labels))
        attn_masks = torch.cat(attn_masks)
        
        eval_data = Eval(predictions=predictions, label_ids=label_ids, attn_masks = attn_masks)
        
        # eval_log = {'predictions': (torch.cat(intent_preds), torch.cat(slot_preds)),
        #              'label_ids': (torch.cat(intent_labels), torch.cat(slot_labels))}

        # with open(os.path.join(os.getcwd(), f'eval_{train_eval}.pkl'), 'wb') as f:
        #     pickle.dump(eval_log, f)
            
        # eval on zh            
        compute_metrics = create_compute_metrics(intent_labels = intent_labels_map, 
                                                 slot_labels = slot_labels_map,
                                                 metrics ='all',)
        res = compute_metrics(eval_data)
        average_loss = step_loss / eval_size

        print(f"Evaluate...\nEval Loss: {average_loss:.4f}\n"
            f"Intent Accuracy: {res['intent_acc']:.2f}\n"
            f"Slot F1: {res['slot_micro_f1']:.2f}\n"
            f"Exact Match Accuracy: {res['ex_match_acc']:.2f}")
        # Logging to file
        log_message = (f"eval loss: {average_loss:.4f}, "
                    f"intent accuracy: {res['intent_acc']:.2f}, "
                    f"slot f1: {res['slot_micro_f1']:.2f}, "
                    f"exact match accuracy: {res['ex_match_acc']:.2f}\n")
        
        lab = ['test','train'][train_eval]
        with open(args.save_dir + f'eval_{lab}.log.pkl', 'a') as f:
            f.write(log_message)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--checkpoint", type = str, default = 'FacebookAI/xlm-roberta-base')
    parser.add_argument("--save_dir", type = str, default = '/scratch/' + os.environ.get("USER", "") + '/out/')
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--label", type=str, default="ICSL")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--debug", action="store_true", help="train a model on the small training data to debug")
    parser.add_argument("--debug_eval", action="store_true", help="evaluate a model on the small training data to debug")
    parser.add_argument("--lang", type = str, default = "zh")
    parser.add_argument("--src", type = str, default = "en")
    parser.add_argument("--epoch", type = int, default = 1)
    parser.add_argument("--average", type = str, default = 'micro')
    args, unknown = parser.parse_known_args()

    random_seed = 1012
    warnings.filterwarnings('ignore')
    torch.seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything()

    base_model = XLMRobertaModel.from_pretrained(args.checkpoint)
    
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    Eval = namedtuple('Eval', ['predictions', 'label_ids', 'attn_masks'])

    en_train = Dataset.from_file('/kaggle/input' + f'/{args.src}-train/data-00000-of-00001.arrow')
    zh_train = Dataset.from_file('/kaggle/input' + f'/{args.lang}-train/data-00000-of-00001.arrow')
        
    zh_val = Dataset.from_file('/kaggle/input' + f'/{args.lang}-dev/data-00000-of-00001.arrow')

    para_dataset = deepcopy(en_train)
    para_dataset = para_dataset.add_column("target_utt", zh_train['utt'])
    para_dataset = para_dataset.add_column("target_slots", zh_train['slots_str'])
    para_dataset = para_dataset.add_column("target_intents", zh_train['intent_str'])
    
    # para_dataset = para_dataset.add_column("target_utt", de_train['utt'])
    # para_dataset = para_dataset.add_column("target_slots", de_train['slots_str'])
    # para_dataset = para_dataset.add_column("target_intents", de_train['intent_str'])
    
    para_dataset = para_dataset.map(lambda x: convert_train(x, src = args.src), batched=True)    

    para_dataloader = DataLoader(para_dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=CollatorMASSIVEIntentClassSlotFill_para(tokenizer=tokenizer, max_length=512))
    train_dataloader = DataLoader(en_train, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length=512))
    eval_dataloader = DataLoader(zh_val, batch_size=args.batch_size, shuffle=True,
                                    collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length=512))
    train_eval_dataloader = DataLoader(zh_train, batch_size=args.batch_size, shuffle=True,
                                    collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length=512))


    # eval_dataloader = DataLoader(de_val, batch_size=args.batch_size, shuffle=True,
    #                                 collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length=512))
    # train_eval_dataloader = DataLoader(de_train, batch_size=args.batch_size, shuffle=True,
    #                                 collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length=512))
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

     # load mappings: should be using en mapping since order matters
    with open('/kaggle/input' + f'/{args.src}-train/{args.src}.intents', 'r', encoding = 'UTF-8') as file:
        intent_labels_map = json.load(file)
    
    with open('/kaggle/input' + f'/{args.src}-train/{args.src}.slots', 'r', encoding = 'UTF-8') as file:
        slot_labels_map = json.load(file)
    
    with open('/kaggle/input'+ f'/{args.lang}-train/{args.lang}.intents', 'r', encoding = 'UTF-8') as file:
        zh_intent_labels_map = json.load(file)
    
    with open('/kaggle/input' + f'/{args.lang}-train/{args.lang}.slots', 'r', encoding = 'UTF-8') as file:
        zh_slot_labels_map =json.load(file)

        
    if args.train:
        # num_slot_labels & num_intents: according to https://arxiv.org/pdf/2204.08582
        # note 56 num_slot_labels! not 55! Original paper was inaccurate.
        model = MultiTaskICSL(base_model, vocab_size, num_slot_labels=56, num_intents=60)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr = args.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 300)

        train(model, optimizer, lr_scheduler, train_dataloader, para_dataloader)
    if args.eval:
        model = MultiTaskICSL(base_model, vocab_size, num_slot_labels=56, num_intents=60)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr= args.lr)
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, args, loader_name='labeled')
        evaluate(model, eval_dataloader, )
    if args.debug:
        model = MultiTaskICSL(base_model, vocab_size, num_slot_labels=56, num_intents=60)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr = args.lr, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 300)
        
        small_para_dataset = para_dataset.shuffle(seed=random_seed).select(range(200))
        small_train_dataset = en_train.shuffle(seed=random_seed).select(range(200))
        small_para_dataloader = DataLoader(small_para_dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=CollatorMASSIVEIntentClassSlotFill_para(tokenizer=tokenizer, max_length=512))
        small_train_dataloader = DataLoader(small_train_dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length=512))
        
        train(model, optimizer, lr_scheduler, small_train_dataloader, small_para_dataloader)

    if args.debug_eval:
        model = MultiTaskICSL(base_model, vocab_size, num_slot_labels=56, num_intents=60)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr= args.lr)
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, args, loader_name='labeled')
        
        small_eval_dataset = zh_val.select(range(10))
        small_eval_dataloader = DataLoader(small_eval_dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length=512))
        evaluate(model, small_eval_dataloader, )