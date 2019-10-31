import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertAdam
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader



class modelDataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, data1, data2, label):
        self.data1 = data1
        self.data2 = data2
        self.label = label
    #返回数据集大小
    def __len__(self):
        return len(self.label)
    #得到数据内容和标签
    def __getitem__(self, index):
        data1 = torch.LongTensor(self.data1[index])
        data2 = torch.LongTensor(self.data2[index])
        label = torch.Tensor([self.label[index]])
        if torch.cuda.is_available():
            data1 = data1.cuda()
            data2 = data2.cuda()
            label = label.cuda()
        return [data1, data2], label 