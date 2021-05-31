import torch
import torch.nn.functional as F
import torch
from torch import nn
import math

from torch.nn.modules.container import ModuleList
from .position import PositionalEncoding
from .encoder_block import EncoderBlock

class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        num_class,
        input_size,  
        len_feature_new=[28, 28, 28], 
        len_seq =300, 
        num_head=3,  
        dropout=0, 
        num_block=3, 
        **kwargs):
        
        super(TransformerEncoder, self).__init__(**kwargs)
        self.len_feature_input= input_size[-1]
        self.len_feature_new = len_feature_new
        self.pos_encoding = PositionalEncoding(self.len_feature_input, dropout)

        self.blks = nn.Sequential()

        for i in range(num_block):
            module =  EncoderBlock(input_size, len_feature_new[i], num_head, dropout)
            self.blks.add_module(str(i), module)
            input_size = (*input_size[:-1], len_feature_new[i])

        self.fc1 = nn.Linear(len_feature_new[-1], num_class*3)
        self.fc2 = nn.Linear(num_class*3, num_class)


    def forward(self, X):
        # X = self.pos_encoding(X * math.sqrt(self.len_feature_input))
        X = self.pos_encoding(X)
        X = self.blks(X)
        X = X.mean(dim=-1)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)

        return X