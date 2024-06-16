
import sys
sys.path.append('xx')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

from transformers import AutoTokenizer, AutoModel,AutoModelForSeq2SeqLM

class Model_T5(nn.Module):
	def __init__(self,model_name):
		super(Model_T5, self).__init__()
		self.t5_decoder = AutoModelForSeq2SeqLM.from_pretrained(model_name)

	def forward(self, input_ids=None,input_embeds = None,attention_mask =None, labels=None):
		# return self.t5_decoder(input_ids = input_ids,inputs_embeds = input_embeds,labels=labels).loss
		return self.t5_decoder(input_ids = input_ids,inputs_embeds = input_embeds,attention_mask = attention_mask,labels=labels).loss
