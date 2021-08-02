import torch
from torch import nn
from transformers import *

class RobertaForAIViVN(BertPreTrainedModel):
   config_class = RobertaConfig
   base_model_prefix = "roberta"
   def __init__(self, config):
       super(RobertaForAIViVN, self).__init__(config)
       self.num_labels = config.num_labels
       self.roberta = RobertaModel(config)
       #self.qa_outputs = nn.Linear(4*config.hidden_size, self.num_labels)
       #self.qa_outputs = nn.Linear(5*config.hidden_size+768, self.num_labels)
       self.qa_outputs = nn.Linear(2*config.hidden_size+768, self.num_labels)

       self.init_weights()

   def forward(self, input_ids, senti_batch, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
   #def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                start_positions=None, end_positions=None):

       outputs = self.roberta(input_ids,
                            attention_mask=attention_mask,
#                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
       #cuda0 = torch.device('cuda:0')
       #senti = torch.ones(24,768 , device=cuda0)
       #senti = torch.tensor([1,2], dtype=torch.float64, device=cuda0)
       #print("Shape ",outputs[2][-1][:,0, ...].size())
       #cls_output = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...],outputs[2][-4][:,0, ...], outputs[2][-5][:,0, ...], senti_batch),-1)
       #cls_output = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...],outputs[2][-4][:,0, ...]),-1)
       cls_output = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], senti_batch),-1)
       logits = self.qa_outputs(cls_output)
       return logits
