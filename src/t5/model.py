import torch
import torch.nn as nn


class Classifier(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder, tokenizer, args, num_labels):
        super(Model, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = Classifier(encoder.config, 2)
        self.type_classifier = Classifier(encoder.config, num_labels)
        self.args = args
    
    def forward(self, input_ids, labels=None, type_labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)).last_hidden_state
        logits = self.classifier(outputs)
        type_logits = self.type_classifier(outputs)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            type_loss = loss_fct(type_logits, type_labels)
            return loss, type_loss
        else:
            prob = torch.softmax(logits, dim=-1)
            type_prob = torch.softmax(type_logits, dim=-1)
            return prob, type_prob