import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam
import numpy as np
from tqdm import tqdm

class matchingModel(nn.Module):
    def __init__(self, pre_trained_model):
        super(matchingModel, self).__init__()
        
        self.model = BertModel.from_pretrained(pre_trained_model)
        self.fc_query = nn.Linear(768, 384)
        self.fc_bidword = nn.Linear(768, 384)
        self.dropout = nn.Dropout(p = 0.2)
        self.fc1 = nn.Linear(384*4, 256)
        self.fc2 = nn.Linear(256, 1) 
        
    def infer_metrics(self, x1, x2):
        return torch.cat((x1, x2, torch.abs(x1 - x2), x1 * x2), 1)
    
    def forward(self, input_data):
        
        query = self.model(input_data[0])
        bidword = self.model(input_data[1])
        query_vec = torch.tanh(self.fc_query(query[1]))
        bidword_vec = torch.tanh(self.fc_bidword(bidword[1]))
        similarity_scores = self.infer_metrics(query_vec, bidword_vec)

        #global average pooling and global max pooling
        # query_vec = torch.tanh(self.fc_query(query[0][0]))
        # bidword_vec = torch.tanh(self.fc_bidword(bidword[0][0]))
        # avg_pool_query = torch.mean(query_vec, 1)
        # avg_pool_bidword = torch.mean(bidword_vec, 1)
        # max_pool_query, _ = torch.max(query_vec, 1)
        # max_pool_bidword, _ = torch.max(bidword_vec, 1)
        # similarity_scores = self.infer_metrics(max_pool_query, max_pool_bidword)

        
        similarity_scores = torch.tanh(self.fc1(similarity_scores))
        similarity_scores = self.fc2(similarity_scores)
        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # output = cos(query_vec[1], bidword_vec[1])
        
        return similarity_scores