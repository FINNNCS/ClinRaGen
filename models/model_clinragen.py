import os, sys
sys.path.insert(0, os.path.abspath("xx"))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from model_patchTST import PatchTSTEncoder
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import json

import math

class MEDTS(nn.Module):
    def __init__(self,model_name, prompt_length, knwoledge_asinput = False, Freeze_t5coder = True, Freeze_TST = True,n_tokens = 1000, n_heads = 8, prompt_encoder_hidden_size = 768,enc_dim = 512, num_features = 11,patch_len = 8, num_patch = 125, stride = 8):
        super(MEDTS, self).__init__()
        self.ts_encoder = PatchTSTEncoder(num_features,prompt_encoder_hidden_size,num_patch,patch_len)
        self.t5_decoder =  AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if Freeze_t5coder:
            for name, param in self.t5_decoder.named_parameters():
                param.requires_grad = False
            self.t5_decoder.eval()
            print("freeze T5")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lab_proj = nn.Linear(prompt_encoder_hidden_size, self.t5_decoder.config.hidden_size)
        self.init_prompt_value = self.t5_decoder.encoder.embed_tokens.weight
        self.mapping_layer = nn.Linear(self.init_prompt_value.shape[0], n_tokens)
        # self.puzzle_token = [0, 1, 27137, 3, 6, 7, 8, 9, 11, 526, 15, 17, 18, 19, 21010, 29, 26653, 35, 9773, 6190, 6189, 1080, 13369, 16443, 60, 63, 66, 8264, 4169, 2632, 80, 2641, 88, 97, 102, 107, 27757, 630, 27255, 1146, 2176, 1666, 6786, 648, 145, 662, 152, 6808, 9378, 13492, 1717, 21182, 192, 1227, 19662, 729, 12521, 12010, 235, 782, 1296, 11035, 31006, 324, 1364, 2388, 2391, 30552, 2407, 874, 1389, 1408, 386, 6546, 17310, 20394, 18358, 3555, 22513, 509]
        kg = json.load(open("xxx/knowledgegraph.json"))
        self.kg =[i.replace("' ","'") for i in list(kg.values())]
        
        puzzle_token = list(set([tk for tks in self.tokenizer(self.kg)["input_ids"] for tk in tks]))
        print("Puzzle token length: ",len(puzzle_token))
        puzzle_set = []
        for idx in puzzle_token:
            puzzle_set.append( self.t5_decoder.encoder.embed_tokens.weight[idx].clone().detach().unsqueeze(0))
        self.puzzle_embedding = torch.cat(puzzle_set,axis = 0)
        # self.puzzle_embedding =  self.t5_decoder.encoder.embed_tokens.weight.clone().detach()

        self.knwoledge_asinput = knwoledge_asinput
        # self.puzzle_prompt = nn.Embedding(self.puzzle_embedding.shape[0], self.ts_decoder.config.hidden_size)
        # self.puzzle_prompt.weight = nn.parameter.Parameter(self.puzzle_embedding)


     
        self.mapping_layer = nn.Linear(self.puzzle_embedding.shape[0],prompt_length)
        self.reprogramming_layer = ReprogrammingLayer(self.t5_decoder.config.hidden_size, n_heads, 32, self.t5_decoder.config.hidden_size)
        if Freeze_TST:
            for name, param in self.ts_encoder.named_parameters():
                param.requires_grad = False
            self.ts_encoder.eval()
            # self.mapping_layer.requires_grad = False
            # self.lab_proj.requires_grad = False
            # for name, param in self.reprogramming_layer.named_parameters():
            #     param.requires_grad = False
            # self.reprogramming_layer.eval()
            print("freeze TST")  

    def forward(self,lab_x = None,label_input = None,text_inputt5 = None,labdecsp_x = None):
        lab_feats = self.ts_encoder(lab_x)
        lab_feats = self.lab_proj(lab_feats)
        # source_embeddings = self.mapping_layer(self.init_prompt_value.permute(1, 0)).permute(1, 0)

        source_embeddings = self.mapping_layer(self.puzzle_embedding.permute(1, 0).to(lab_feats.device)).permute(1, 0)
        lab_feats = self.reprogramming_layer(lab_feats, source_embeddings, source_embeddings)
        lab_feats_mask = torch.ones(lab_feats.size()[:-1], dtype=torch.long).to(lab_feats.device)
        if self.knwoledge_asinput:
            kg_token = self.tokenizer(self.kg, return_tensors="pt",padding=True).to(lab_feats.device)
            kg_embedding = self.t5_decoder.encoder.embed_tokens(kg_token["input_ids"])[:,0,:].repeat(lab_feats.shape[0],1,1)
            kg_mask = torch.ones(kg_embedding.size()[:-1], dtype=torch.long).to(kg_embedding.device)
            mm_input = torch.cat((self.t5_decoder.encoder.embed_tokens(text_inputt5["input_ids"]),lab_feats,kg_embedding),axis = 1)
            mm_mask = torch.cat((text_inputt5["attention_mask"],lab_feats_mask,kg_mask),axis = 1)
        else:
            mm_input = torch.cat((self.t5_decoder.encoder.embed_tokens(text_inputt5["input_ids"]),lab_feats),axis = 1)
            mm_mask = torch.cat((text_inputt5["attention_mask"],lab_feats_mask),axis = 1)

        output = self.t5_decoder(inputs_embeds=mm_input,attention_mask = mm_mask, labels=label_input["input_ids"])
        loss_gen = output.loss

        return loss_gen,mm_input
    
class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
