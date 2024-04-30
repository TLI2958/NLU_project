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

# From MASSIVE, slight modifications: 
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

import json
import numpy as np
import sklearn.metrics as sklm
from seqeval.metrics import f1_score
from math import sqrt

# note: we are using the dataset: 
# https://huggingface.co/datasets/AmazonScience/massive

# collator function for MASSIVE intent classification and slot filling
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

    def __init__(self, tokenizer, max_length, padding='longest', pad_to_multiple_of=None):
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
        return {k: torch.tensor(v, dtype=torch.int64) for k, v in pad_tok_inputs.items()}

    
# compute metrics
def create_compute_metrics(intent_labels, slot_labels, tokenizer=None, ignore_labels=None,
                           metrics='all'):
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

    # COLLATOR: MASSIVE INTENT CLASS SLOT FILL
    def compute_metrics(p):
        # p is named tuple with `predictions` and `label_ids`.
        # p.predictions is a tuple of two elements, the first being the intent classification
        # predictions of size num_examples and the second being the slot classification preds
        # of size num_examples. Each intent classification pred is of size num_intent_classes,
        # and each slot classification prediction is of shape (seq_len, num_slot_classes)
        # label_ids is tuple of two elements, first array of all IC labels (size num_examples)
        # The second element is size num_examples with each entry sized seq_len x num_slot_class

        intent_preds = p.predictions[0]
        slot_preds = p.predictions[1]

        intent_label_tuple = p.label_ids[0]
        slot_label_tuple = p.label_ids[1]

        intent_preds_am = [np.argmax(x) for x in intent_preds]
        slot_preds_am = [np.argmax(x, axis=1) for x in slot_preds]

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
            pad='Other'
        )


    return compute_metrics


# eval function: 
def eval_preds(pred_intents=None, lab_intents=None, pred_slots=None, lab_slots=None,
               eval_metrics='all', labels_ignore='Other', labels_merge=None, pad='Other'):
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
        for lab, pred in zip(lab_slots, pred_slots):

            # Pad or truncate prediction as needed using `pad` arg
            if type(pred) == list:
                pred = pred[:len(lab)] + [pad]*(len(lab) - len(pred))

            # Fix for Issue 21 -- subwords after the first one from a word should be ignored
            for i, x in enumerate(lab):
                if x == -100:
                    pred[i] = -100

            # convert to BIO
            bio_slot_labels.append(
                convert_to_bio(lab, outside=labels_ignore, labels_merge=labels_merge)
            )
            bio_slot_preds.append(
                convert_to_bio(pred, outside=labels_ignore, labels_merge=labels_merge)
            )

    if ('slot_micro_f1' in eval_metrics) or ('all' in eval_metrics):

        # from seqeval
        smf1 = f1_score(bio_slot_labels, bio_slot_preds)
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

def convert_to_bio(seq_tags, outside='Other', labels_merge=None):
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

    seq_tags = [str(x) for x in seq_tags]

    outside = [outside] if type(outside) != list else outside
    outside = [str(x) for x in outside]

    if labels_merge:
        labels_merge = [labels_merge] if type(labels_merge) != list else labels_merge
        labels_merge = [str(x) for x in labels_merge]
    else:
        labels_merge = []

    bio_tagged = []
    prev_tag = None
    for tag in seq_tags:
        if prev_tag == None and tag in labels_merge:
            bio_tagged.append('O')
        elif tag in outside:
            bio_tagged.append('O')
            prev_tag = tag
        elif tag != prev_tag and tag not in labels_merge:
            bio_tagged.append('B-' + tag)
            prev_tag = tag
        elif tag == prev_tag or tag in labels_merge:
            if prev_tag in outside:
                bio_tagged.append('O')
            else:
                bio_tagged.append('I-' + prev_tag)

    return bio_tagged

def eval_preds(pred_intents=None, lab_intents=None, pred_slots=None, lab_slots=None,
               eval_metrics='all', labels_ignore='Other', labels_merge=None, pad='Other'):
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
        for lab, pred in zip(lab_slots, pred_slots):

            # Pad or truncate prediction as needed using `pad` arg
            if type(pred) == list:
                pred = pred[:len(lab)] + [pad]*(len(lab) - len(pred))

            # Fix for Issue 21 -- subwords after the first one from a word should be ignored
            for i, x in enumerate(lab):
                if x == -100:
                    pred[i] = -100

            # convert to BIO
            bio_slot_labels.append(
                convert_to_bio(lab, outside=labels_ignore, labels_merge=labels_merge)
            )
            bio_slot_preds.append(
                convert_to_bio(pred, outside=labels_ignore, labels_merge=labels_merge)
            )

    if ('slot_micro_f1' in eval_metrics) or ('all' in eval_metrics):

        # from seqeval
        smf1 = f1_score(bio_slot_labels, bio_slot_preds)
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

def output_predictions(outputs, intent_labels, slot_labels, conf, tokenizer=None,
                       combine_slots=True, remove_slots=None, add_pred_parse=True,
                       save_to_file=True):
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

    pred_file = conf.get('train_val.predictions_file')

    if pred_file and (conf.get('train_val.trainer_args.locale_eval_strategy') != 'all only'):
        raise NotImplementedError("You must use 'all only' as the locale_eval_strategy if you"
                                  " specify a predictions file")

    final_outputs = []

    # if there is a space within sequence of subwords that should be joined back together,
    # it's probably because the tokenizer converted a Zero Width Space to a normal space.
    # Make this False to not replace the space with a ZWSP when re-joining subwords
    replace_zwsp = conf.get('test.replace_inner_space_zwsp', default=True)

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


# checkpoint
def save_checkpoint(model, optimizer, epoch, i, args, loader_name = None):
    if loader_name is None:
        checkpoint_path = f'{args.save_dir}/trained_{args.label}_checkpoint.pth'
        torch.save({
            'epoch': epoch,
            'iteration': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
    else:
        checkpoint_path = f'{args.save_dir}/trained_{args.label}_{loader_name}_checkpoint.pth'
        torch.save({
            'epoch': epoch,
            'iteration': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

    
def load_checkpoint(model, optimizer, args, loader_name = None):
    if loader_name is None:
        checkpoint_path = f'{args.save_dir}/trained_{args.label}_checkpoint.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            print(f"Checkpoint found. Resuming training from epoch {epoch}, iteration {iteration}.")
            return model, optimizer, epoch, iteration
        else:
            return model, optimizer, 0, 0
    else:
        checkpoint_path = f'{args.save_dir}/trained_{args.label}_{loader_name}_checkpoint.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            iteration = checkpoint['iteration']
            print(f"Checkpoint found. Resuming training from epoch {epoch}, iteration {iteration}.")
            return model, optimizer, epoch, iteration
        else:
            return model, optimizer, 0, 0