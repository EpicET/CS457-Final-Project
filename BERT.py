
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class ApplicationReviewModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert =  AutoModel.from_pretrained('distilbert-base-uncased')
        self.linear_layer = nn.Linear(768, 6)
        

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  
        logits = self.linear_layer(cls_embedding)
        return logits 

