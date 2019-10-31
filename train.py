import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam, BertConfig, convert_tf_checkpoint_to_pytorch
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
from utils import modelDataset
from model import matchingModel
import argparse 

def get_args():
    parser = argparse.ArgumentParser("MachingModel")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--model_path", type=str, default='../bert_model/uncased_L-12_H-768_A-12/')
    parser.add_argument("--max_sequence_length", type=int, default=200) 
    parser.add_argument("--hidden_dim", type=int, default=768) 
    parser.add_argument("--log_path", type=str, default="result/log_data.txt")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_model", type=str, default=None) 
    args = parser.parse_args() 
    return args 

def convert_lines(sentence, max_seq_length, tokenizer):
    max_seq_length -=2
    all_tokens = []
    for text in tqdm(sentence):
        tokens = tokenizer.tokenize(text)
        if len(tokens)>max_seq_length:
            tokens = tokens[:max_seq_length]
        padding_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens+["[SEP]"])+[0] * (max_seq_length - len(tokens))
        all_tokens.append(padding_token)
    return np.array(all_tokens)

def main(args):
    

    batch_size = args.batch_size
    lr = args.lr
    num_epochs = args.num_epochs
    max_sequence_length = args.max_sequence_length
    BERT_MODEL_PATH = args.model_path
    hidden_dim = args.hidden_dim
    bert_config = BertConfig(BERT_MODEL_PATH+'bert_config.json')
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None, do_lower_case=True)
    # convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
    #         BERT_MODEL_PATH + 'bert_model.ckpt', 
    #         BERT_MODEL_PATH + 'bert_config.json',
    #         BERT_MODEL_PATH + 'pytorch_model.bin')

    file = pd.read_csv('quora.tsv', '\t')[:10]
    #file['question1'] = file['question1'].astype(str) 
    question1 = convert_lines(file["question1"], max_sequence_length, tokenizer)
    #file['question2'] = file['question2'].astype(str)
    question2 = convert_lines(file["question2"], max_sequence_length, tokenizer)
    label = np.array(file['is_duplicate']) 
    dataset = modelDataset(question1, question2, label)

    model = matchingModel(BERT_MODEL_PATH, hidden_dim)
    model.train() 

    accumulation_steps=2
    save_steps = 1000
    checkpoint = None 
    num_train_optimization_steps = int(num_epochs*len(label) / batch_size / accumulation_steps)
    optimizer = BertAdam(model.parameters(),
                        lr=lr,  
                        warmup=0.05,   
                        t_total=num_train_optimization_steps)

    criterion = nn.BCEWithLogitsLoss()
    for epoch in tqdm(range(num_epochs)): 
    #     file_name = 'loss_log_' + 'epoch' + str(epoch) + '.txt'
    #     file = open(file_name, 'w', encoding='utf-8')
        train_loader = DataLoader.DataLoader(dataset, batch_size= batch_size, shuffle = True) 
        avg_loss = 0
        avg_accuracy = 0
        optimizer.zero_grad()   
        for x_batch, y_batch in train_loader:
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)  
            loss.backward()  
            optimizer.step() 
            optimizer.zero_grad()
            avg_loss += loss.item() / len(train_loader)
            avg_accuracy += torch.mean(((torch.sigmoid(y_pred)>0.5) == (y_batch>0.5)).to(torch.float) ).item()/len(train_loader)
    #         i += 1
    #         file.write('batch' + str(i) + '\t' + 'avg_loss' + '=' + str(avg_loss) + '\t' + 'avg_accuracy' + '=' + str(avg_accuracy) + '\n')
    #     file.close()
    #     file_path = output_model_file + str(epoch) +'.bin'
    #     torch.save(model.state_dict(), file_path)
        print(f'| Epoch: {epoch+1:02} | Train Loss: {avg_loss:.3f} | Train Acc: {avg_accuracy*100:.2f}%')
   
if __name__ == "__main__":
    args = get_args()
    main(args)

