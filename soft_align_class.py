
# modified from soft-align implementation in multiatis: 
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
from torch.utils.data import DataLoader


from transformers import AutoTokenizer, XLMRobertaModel
from datasets import load_dataset, Dataset


class MultiTaskICSL(nn.Module):
    """Model for IC/SL task.

    The model feeds token ids into XLM-RoBERTa to get the sequence
    representations, then apply two dense layers for IC/SL task.
    """

    def __init__(self, base_model, vocab_size, num_slot_labels, num_intents, hidden_size=768, 
                 dropout=.1, prefix=None, params=None):
        super(MultiTaskICSL, self).__init__(prefix=prefix, params=params)
        self.base_model = base_model
        with self.name_scope():
            self.dropout = nn.Dropout(rate=dropout)
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
            self.ffn_layer = nn.Sequential(nn.LayerNorm(hidden_size),
                                            nn.Linear(hidden_size, hidden_size),
                                            nn.Dropout(.02),
                                            nn.Linear(hidden_size, hidden_size*2),)
   

    def encode(self, inputs, attn_mask = None):
        encoded = self.base_model(input_ids = inputs, attention_mask = attn_mask)
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
        hidden = self.encode(input_ids = inputs, attention_mask = attn_mask)
        
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
        tgt_embed = self.base_model.embedding(target)

        attn_output, _ = self.attention_layer(tgt_embed, src_encoded, source_attn_mask)
        decoded = self.ffn_layer(attn_output)

        # source reconstruction 
        translation = self.lm_output_layer(decoded[:, 1:, :])

        # IC and SL
        intent_prediction = self.intent_classifier(src_encoded[:, 0, :])
        slot_prediction = self.slot_classifier(attn_output[:, 1:, :])
        return translation, intent_prediction, slot_prediction