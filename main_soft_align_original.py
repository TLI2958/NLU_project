"""
This version swaps the designation of source and target, as per paper. 
"""


# heavily modified from soft-align implementation in multiatis: 
# https://github.com/amazon-science/multiatis/blob/main/code/scripts/bert_soft_align.py
# implementation of soft-align in paper: https://arxiv.org/pdf/2004.14353
# original code is in mxnet, this is a port to pytorch + transformers

# also from MASSIVE utils:
# https://github.com/alexa/massive/blob/main/src/massive/utils/

from utils import *
from soft_align_class import *
from torch.cuda.amp import autocast, GradScaler

def seed_everything(seed=random_seed):
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
    sl_loss_fn = nn.CrossEntropyLoss(reduction='none')
    mt_loss_fn = nn.CrossEntropyLoss(reduction='none')

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
            sl_loss = sl_loss * source_attn_mask[:, 1:]
            sl_loss = (sl_loss.sum(-1)/source_attn_mask[:, 1:].sum(-1)).mean()

            mce_loss = mt_loss_fn(translation.transpose(1,2), target[:, 1:],) # since zh-en
            mce_loss = mce_loss * source_attn_mask[:, 1:]
            mce_loss = (mce_loss.sum(-1)/source_attn_mask[:, 1:].sum(-1)).mean()

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
            sl_loss = sl_loss_fn(slot_pred.transpose(1,2), slot_label[:, 1:])
            sl_loss = sl_loss * attn_mask[:, 1:]
            sl_loss = (sl_loss.sum(-1)/attn_mask[:, 1:].sum(-1)).mean()

            loss = ic_loss + sl_loss.mean()
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
    sl_loss_fn = nn.CrossEntropyLoss(reduction='none')

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
            sl_loss = sl_loss_fn(slot_pred.transpose(1,2), slot_label[:, 1:])
            sl_loss = sl_loss * attn_mask[:, 1:]
            sl_loss = (sl_loss.sum(-1)/attn_mask[:, 1:].sum(-1)).mean()

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
                                                 metrics ='all',
                                                 ignore_labels= ['Other']
                                                 )
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

    args = parser.parse_args()
    random_seed = 1012
    warnings.filterwarnings('ignore')
    torch.seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything()

    base_model = XLMRobertaModel.from_pretrained(args.checkpoint)
    
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    Eval = namedtuple('Eval', ['predictions', 'label_ids', 'attn_masks'])

    en_train = Dataset.from_file(os.getcwd() + f'/data_{args.src}/{args.src}.train/data-00000-of-00001.arrow')
    zh_train = Dataset.from_file(os.getcwd() + f'/data_{args.lang}/{args.lang}.train/data-00000-of-00001.arrow')
        
    zh_val = Dataset.from_file(os.getcwd() + f'/data_{args.lang}/{args.lang}.dev/data-00000-of-00001.arrow')

    para_dataset = deepcopy(en_train)
    para_dataset = para_dataset.add_column("target_utt", zh_train['utt'])
    # not used
    para_dataset = para_dataset.add_column("target_slots", zh_train['slots_str'])
    para_dataset = para_dataset.add_column("target_intents", zh_train['intent_str'])
    
    # para_dataset = para_dataset.add_column("target_utt", de_train['utt'])
    # para_dataset = para_dataset.add_column("target_slots", de_train['slots_str'])
    # para_dataset = para_dataset.add_column("target_intents", de_train['intent_str'])
    
    para_dataset = para_dataset.map(lambda x: convert_train(x, src = args.src), batched=True)    

    para_dataloader = DataLoader(para_dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=CollatorMASSIVEIntentClassSlotFill_para(tokenizer=tokenizer, max_length =200))
    train_dataloader = DataLoader(en_train, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length =200))
    eval_dataloader = DataLoader(zh_val, batch_size=args.batch_size, shuffle=True,
                                    collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length =200))
    train_eval_dataloader = DataLoader(zh_train, batch_size=args.batch_size, shuffle=True,
                                    collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length =200))


    # eval_dataloader = DataLoader(de_val, batch_size=args.batch_size, shuffle=True,
    #                                 collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length =100))
    # train_eval_dataloader = DataLoader(de_train, batch_size=args.batch_size, shuffle=True,
    #                                 collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length =100))
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)

     # load mappings: should be using en mapping since order matters
    with open(os.getcwd() + f'/data_{args.src}/{args.src}.intents', 'r', encoding = 'UTF-8') as file:
        intent_labels_map = json.load(file)
    
    with open(os.getcwd() + f'/data_{args.src}/{args.src}.slots', 'r', encoding = 'UTF-8') as file:
        slot_labels_map = json.load(file)
    
    with open(os.getcwd() + f'/data_{args.lang}/{args.lang}.intents', 'r', encoding = 'UTF-8') as file:
        zh_intent_labels_map = json.load(file)
    
    with open(os.getcwd() + f'/data_{args.lang}/{args.lang}.slots', 'r', encoding = 'UTF-8') as file:
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
        
        small_para_dataset = para_dataset.shuffle(seed=random_seed).select(range(1000))
        small_train_dataset = en_train.shuffle(seed=random_seed).select(range(1000))
        small_para_dataloader = DataLoader(small_para_dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=CollatorMASSIVEIntentClassSlotFill_para(tokenizer=tokenizer, max_length =200))
        small_train_dataloader = DataLoader(small_train_dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length =200))
        
        train(model, optimizer, lr_scheduler, small_train_dataloader, small_para_dataloader)

    if args.debug_eval:
        model = MultiTaskICSL(base_model, vocab_size, num_slot_labels=56, num_intents=60)
        model = model.to(device)
        optimizer = Adam(model.parameters(), lr= args.lr)
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, args, loader_name='labeled')
        
        small_eval_dataset = zh_val.select(range(10))
        small_eval_dataloader = DataLoader(small_eval_dataset, batch_size=args.batch_size, shuffle=True, 
                                collate_fn=CollatorMASSIVEIntentClassSlotFill(tokenizer=tokenizer, max_length =200))
        evaluate(model, small_eval_dataloader, )


