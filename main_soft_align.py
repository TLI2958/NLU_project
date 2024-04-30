# modified from soft-align implementation in multiatis: 
# https://github.com/amazon-science/multiatis/blob/main/code/scripts/bert_soft_align.py
# implementation of soft-align in paper: https://arxiv.org/pdf/2004.14353
# original code is in mxnet, this is a port to pytorch + transformers

# also from MASSIVE utils:
# https://github.com/alexa/massive/blob/main/src/massive/utils/


from collections import namedtuple
import os
from tqdm import tqdm
import numpy as np
import random
import argparse
import warnings

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from transformers import AutoTokenizer, XLMRobertaModel
from datasets import load_dataset
import evaluate

from massive_utils import *

random_seed = 1012
warnings.filterwarnings('ignore')

torch.seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=random_seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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

def train(train_dataloader, para_dataloader):
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    # num_slot_labels & num_intents: according to https://arxiv.org/pdf/2204.08582
    
    model = MultiTaskICSL(base_model, vocab_size, num_slot_labels=55, num_intents=60) 
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, args)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ic_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    sl_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    mt_loss_fn = nn.CrossEntropyLoss(reduction='mean')

    paral_size = len(para_dataloader)
    label_size = len(train_dataloader)

    #Eval = namedtuple('Eval', ['predictions', 'label_ids'])

    pbar = tqdm(range(max(1, start_epoch + 1), args.num_epochs + 1))
    model.train()

    for epoch in pbar:
        mt_loss, icsl_loss, step_loss = 0, 0, 0
        
        # prepare for eval
        intent_preds = []
        slot_preds = []
        intent_labels = []
        slot_labels = []
        
        # train on parallel data
        for i, para_batch in tqdm(enumerate(para_dataloader), 
                                  total=len(para_dataloader)):
            source, target, slot_label, intent_label, source_attn_mask = para_batch
            source, target, slot_label, intent_label, source_attn_mask = (source.to(device), 
                                                                         target.to(device), slot_label.to(device), 
                                                                         intent_label.to(device), 
                                                                         source_attn_mask.to(device))
            
            translation, intent_pred, slot_pred = model.translate_and_predict(source, 
                                                                              target, source_attn_mask)
            intent_preds.append(intent_pred)
            slot_preds.append(slot_pred)
            intent_labels.append(intent_label)
            slot_labels.append(slot_label)

            ic_loss = ic_loss_fn(intent_pred, intent_label)
            sl_loss = sl_loss_fn(slot_pred.transpose(1,2), slot_label)
            mce_loss = mt_loss_fn(translation.transpose(1,2), target[:, 1:])
            loss = ic_loss + sl_loss + mce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            icsl_loss += ic_loss.item() + sl_loss.item()
            mt_loss += mce_loss.item()
        

        save_checkpoint(model, optimizer, epoch, args, load_dataset = 'parallel')
        
        predictions = (torch.cat(intent_preds), torch.cat(slot_preds))
        label_ids = (torch.cat(intent_labels), torch.cat(slot_labels))
        eval_data = Eval(predictions=predictions, label_ids=label_ids)
        compute_metrics = create_compute_metrics()
        with torch.no_grad():
            res = compute_metrics(eval_data)
        print('training on parallel data...')
        pbar.set_postfix({'dataset': 'parallel',
                        'train_loss': loss.item() / paral_size , 
                          'icsl_loss': icsl_loss / paral_size,
                          'mt_loss': mt_loss / (paral_size ),
                          'intent_acc': res['intent_acc'],
                          'slot_f1': res['slot_f1'],
                          'ex_match_acc': res['ex_match_acc']})
        

        # train on labeled data
        intent_preds = []
        slot_preds = []
        intent_labels = []
        slot_labels = []


        for i, batch in tqdm(enumerate(train_dataloader), 
                             total=len(train_dataloader)):
            inputs, slot_label, intent_label, attn_mask = batch
            inputs, slot_label, intent_label, attn_mask = (inputs.to(device), slot_label.to(device), 
                                                           intent_label.to(device), attn_mask.to(device))
            intent_pred, slot_pred = model(inputs, attn_mask)
            ic_loss = ic_loss_fn(intent_pred, intent_label)
            sl_loss = sl_loss_fn(slot_pred, slot_label)
            loss = ic_loss + sl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step_loss += loss.item()

            intent_preds.append(intent_pred)
            slot_preds.append(slot_pred)
            intent_labels.append(intent_label)
            slot_labels.append(slot_label)


            if i%200 == 0:
                save_checkpoint(model, optimizer, epoch, i, args, load_dataset = 'labeled')        

        predictions = (torch.cat(intent_preds), torch.cat(slot_preds))
        label_ids = (torch.cat(intent_labels), torch.cat(slot_labels))
        eval_data = Eval(predictions=predictions, label_ids=label_ids)
        with torch.no_grad():
            compute_metrics = create_compute_metrics()
        res = compute_metrics(eval_data)
        print('training on labeled data only...')
        pbar.set_postfix({'dataset': 'labeled',
                          'train_loss': loss.item() / (label_size),
                          'intent_acc': res['intent_acc'],
                          'slot_f1': res['slot_f1'],
                          'ex_match_acc': res['ex_match_acc']})
        
        with open(os.path.join(args.save_dir, 'train.log'), 'a') as f:
            f.write(f'epoch: {epoch}\ticsl_loss: {icsl_loss / paral_size}\tmt_loss: {mt_loss / paral_size}\tstep_loss: {step_loss / label_size}\n\nintent_acc: {res["intent_acc"]}\tslot_f1: {res["slot_f1"]}\tex_match_acc: {res["ex_match_acc"]}\n')



def evaluate(model, eval_dataloader):
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
    #output_to_eval = namedtuple('Eval', ['predictions', 'label_ids'])

    eval_size = len(eval_dataloader)    

    model.to(device)
    model.eval()
    with torch.no_grad():
        step_loss = 0
        for i, batch in tqdm(enumerate(eval_dataloader), 
                                total=len(eval_dataloader)):
            inputs, slot_label, intent_label, attn_mask = batch
            inputs, slot_label, intent_label, attn_mask = (inputs.to(device), slot_label.to(device), 
                                                            intent_label.to(device), attn_mask.to(device))
            intent_pred, slot_pred = model(inputs, attn_mask)
            ic_loss = ic_loss_fn(intent_pred, intent_label)
            sl_loss = sl_loss_fn(slot_pred, slot_label)
            loss = ic_loss + sl_loss
            step_loss += loss.item()

            intent_preds.append(intent_pred)
            slot_preds.append(slot_pred)
            intent_labels.append(intent_label)
            slot_labels.append(slot_label)     

        predictions = (torch.cat(intent_preds), torch.cat(slot_preds))
        label_ids = (torch.cat(intent_labels), torch.cat(slot_labels))
        eval_data = Eval(predictions=predictions, label_ids=label_ids)
        compute_metrics = create_compute_metrics()
        res = compute_metrics(eval_data)
        average_loss = step_loss / eval_size

        print(f"Evaluate...\nEval Loss: {average_loss:.4f}\n"
            f"Intent Accuracy: {res['intent_acc']:.2f}\n"
            f"Slot F1: {res['slot_f1']:.2f}\n"
            f"Exact Match Accuracy: {res['ex_match_acc']:.2f}")
        # Logging to file
        log_message = (f"eval loss: {average_loss:.4f}, "
                    f"intent accuracy: {res['intent_acc']:.2f}, "
                    f"slot f1: {res['slot_f1']:.2f}, "
                    f"exact match accuracy: {res['ex_match_acc']:.2f}\n")
            
        with open(os.path.join(args.save_dir, 'eval.log'), 'a') as f:
            f.write(log_message)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--checkpoint", type = str, default = 'facebookAI/xlm-roberta-base')
    parser.add_argument("--save_dir", type = str, default = '/scratch/' + os.environ.get("USER", "") + '/out/')
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    seed_everything()

    base_model = XLMRobertaModel.from_pretrained('facebookAI/xlm-roberta-base')
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    Eval = namedtuple('Eval', ['predictions', 'label_ids'])

    
    

