
import sys
sys.path.append('xxx')
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader.dataloader_mimiciv import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
from transformers import AutoTokenizer
from models.model_medts import MEDTS
from sentence_transformers import SentenceTransformer, util
from accelerate import Accelerator
from transformers import AdamW, get_cosine_schedule_with_warmup
import socket
similar_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

import requests
import torch
import datetime
import dill
import copy
import time
start = time.time()
SEED = 3407 #gpu23 model 2
import json
import getpass
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch", default=12, type = int)
parser.add_argument("--rt", action= "store_true")

args = parser.parse_args()

torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0,1"

# conda activate qformer
model_name = "google/flan-t5-small"
num_epochs = 1000
max_length = 500
BATCH_SIZE = args.batch
patch_len = 8
num_patch = 125
puzzle_length = 24
pretrained = True
SV_WEIGHTS = True
evaluation = False
logs = True
Rationale_ts = args.rt
inlcude_rationale = True
Freeze_t5coder = True
Freeze_TST = False
Best_F1 = 0.2
date = str(datetime.date.today())
save_dir= "xxxx"
sv_model_id = model_name.split("/")[-1]
gpuid = socket.gethostname()
save_name = f"{sv_model_id}_{date}_ts_recon_tsratioanle_len{puzzle_length}_{SEED}_{gpuid}"
log_file_name = f'xxx.txt'

if Rationale_ts:
    weight_dir = "xxx.pth"
else:
    weight_dir = "xxx.pth"


with open(f'{log_file_name}', 'a+') as log_file:
    log_file.write(f"Server: {gpuid}, date: {date}, SEED: {SEED}, model_name: {model_name},puzzle_length: {puzzle_length}, pretrained: {pretrained}, BATCH_SIZE: {BATCH_SIZE}, Rationale_ts: {Rationale_ts}, inlcude_rationale: {inlcude_rationale}, max_length: {max_length}, Freeze_TST: {Freeze_TST} "+'\n')
    log_file.close()
start_epoch = 0
print("#####################")
print(f"Server: {gpuid}, date: {date}, SEED: {SEED}, model_name: {model_name},puzzle_length: {puzzle_length}, pretrained: {pretrained}, BATCH_SIZE: {BATCH_SIZE}, Rationale_ts: {Rationale_ts}, inlcude_rationale: {inlcude_rationale}, max_length: {max_length}, Freeze_t5coder: {Freeze_t5coder},  "+'\n')
print("#####################")

target_diagnosis_name_list = ["Acute and unspecified renal failure",
"Acute cerebrovascular disease",
"Acute myocardial infarction",
"Complications of surgical procedures or medical care",
"Fluid and electrolyte disorders",
"Gastrointestinal hemorrhage",
"Other lower respiratory disease",
"Other upper respiratory disease",
"Pleurisy; pneumothorax; pulmonary collapse",
"Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
"Respiratory failure; insufficiency; arrest (adult)",
"Septicemia (except in labor)",
"Shock",
"Chronic kidney disease",
"Chronic obstructive pulmonary disease and bronchiectasis",
"Coronary atherosclerosis and other heart disease",
"Diabetes mellitus without complication",
"Disorders of lipid metabolism",
"Essential hypertension",
"Hypertension with complications and secondary hypertension",
"Cardiac dysrhythmias",
"Conduction disorders",
"Congestive heart failure; nonhypertensive",
"Diabetes mellitus with complications",
"Other liver diseases",
]


if evaluation:
    pretrained = True
    SV_WEIGHTS = False
    Logging = False
    weight_dir = "xxx.pth"


tokenizer = AutoTokenizer.from_pretrained(model_name)

accelerator = Accelerator(
    mixed_precision="no",
    device_placement=True,
)
device = accelerator.device


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [sq[0].shape[0] for sq in data]
    input_x = [i[0].tolist() for i in data]
    disease = [f"### Disease Diagnosed: {i[1]}" for i in data]
    input_x = rnn_utils.pad_sequence([torch.from_numpy(np.array(x)) for x in input_x],batch_first = True, padding_value=0)

    text = [f"Diagnose disease from the following medical notes: \n{d[2]}\n and Lab Test: \n" for d in data]
    label_list = [i[3] for i in data]
    lab_description = [i[4] for i in data]
    disease_with_rationale = [f"### Disease Diagnosed: {d[1]}. ### Rationale: {d[6]}" for d in data]

    return input_x,disease_with_rationale,text,label_list,lab_description

def fit(epoch,model,dataloader,optimizer,scheduler,flag='train'):
    global Best_F1
    if flag == 'train':
        model.train()

    else:
        model.eval()

    # ts_encoder.to(device)
    batch_loss_list = []
    y_list = []
    pred_list_f1 = []
    model = model.to(device)

    for i,(lab_x,labels,text_list,label_list,lab_description) in enumerate(tqdm(dataloader)):
        # if i == 10:break
        label = torch.tensor(np.array(label_list)).to(torch.float32).to(device)
        if flag == "train": 
            with torch.set_grad_enabled(True):
                lab_x = torch.tensor(lab_x).to(torch.float32).to(device)
                lab_x = lab_x.view(lab_x.shape[0],num_patch,lab_x.shape[-1],patch_len)
                label_input = tokenizer(labels, return_tensors="pt",padding=True).to(device)

                loss,lab_feats =  model(lab_x,label_input)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                lab_x = torch.tensor(lab_x).to(torch.float32).to(device)
                lab_x = lab_x.view(lab_x.shape[0],num_patch,lab_x.shape[-1],patch_len)
                label_input = tokenizer(labels, return_tensors="pt",padding=True).to(device)

                loss,lab_feats =  model(lab_x,label_input)
                output_sequences = model.t5_decoder.generate(
                    inputs_embeds = lab_feats,
                    # input_ids = text_input["input_ids"],
                    num_beams = 1,
                    max_length = 500,
                    temperature = 0.8,
                    num_return_sequences = 1,
                )  
                pred_labels = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                # print(labels)
                # print(pred_labels)
                pred = []
                for pred_label in pred_labels:
                    pred_label = pred_label.split("### Disease Diagnosed: ")[-1]
                    if ". ### Rationale" in pred_label:
                        pred_label = pred_label.split(". ### Rationale: ")[0]
                    # print(pred_label)
                    s_pred = [0]*len(target_diagnosis_name_list)
                    for i,d in enumerate(target_diagnosis_name_list):  
                        if d in pred_label:
                            s_pred[i] = 1  
                    pred.append(s_pred) 

                pred = np.array(pred)   
                # print(pred.shape)
                y = np.array(label.cpu().data.tolist())

                # print("disaese label: ",y)
                # print("disease pred: ",pred)

                # print("..............................")
                y_list.append(y)
                pred_list_f1.append(pred)
                batch_loss_list.append( loss.cpu().data )  


    if flag == "dev":
       
        y_list = np.vstack(y_list).squeeze()
        pred_list_f1 = np.vstack(pred_list_f1).squeeze()
        acc = metrics.accuracy_score(y_list,pred_list_f1)
        precision_micro = metrics.precision_score(y_list,pred_list_f1,average='micro')
        recall_micro =  metrics.recall_score(y_list,pred_list_f1,average='micro')
        precision_macro = metrics.precision_score(y_list,pred_list_f1,average='macro')
        recall_macro =  metrics.recall_score(y_list,pred_list_f1,average='macro')

        f1_micro = metrics.f1_score(y_list,pred_list_f1,average="micro")
        f1_macro = metrics.f1_score(y_list,pred_list_f1,average="macro")
        total_loss = sum(batch_loss_list) / len(batch_loss_list)
        end = time.time()
        running_time = end - start
        print("PHASE: {} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} | ACC: {} | Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,acc,total_loss))

        if logs:
            with open(f'{log_file_name}', 'a+') as log_file:
                log_file.write("PHASE: {} EPOCH : {} | Total LOSS  : {}  ".format(flag,epoch + 1,total_loss)+'\n')
                log_file.close()
        if SV_WEIGHTS:
            if f1_micro > Best_F1:
                Best_F1 = f1_micro
                PATH=f"{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(total_loss),4)}_f1_micro_{round(float(f1_micro),4)}_f1_macro_{round(float(f1_macro),4)}.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)



if __name__ == '__main__':

    train_dataset = PatientDataset('mimicdata/mimic4/', flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    dev_dataset = PatientDataset('mimicdata/mimic4/', flag="dev")
    devloader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)

    print(train_dataset.__len__())
    print(dev_dataset.__len__())
    model = MEDTS(puzzle_length,model_name, Freeze_t5coder = Freeze_t5coder, Freeze_TST = Freeze_TST)

    if pretrained:
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device)), strict=False)
        print("loading weight: ",weight_dir)

    optimizer = AdamW(model.parameters(True), lr=5e-5, eps = 1e-8, weight_decay = 0.05)

    len_dataset = train_dataset.__len__()
    total_steps = (len_dataset // BATCH_SIZE) * 100 if len_dataset % BATCH_SIZE == 0 else (len_dataset // BATCH_SIZE + 1) * num_epochs 
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = 200, num_training_steps = total_steps)

    model, optimizer,scheduler, train_dataloader, devloader = accelerator.prepare(model, optimizer,scheduler, trainloader, devloader)
    print("model param numbers: ",sum(p.numel() for p in model.parameters()))

    if evaluation:
    
        fit(1,model,devloader,optimizer,scheduler,flag='dev')
     
    else:
        for epoch in range(start_epoch,num_epochs):

            fit(epoch,model,trainloader,optimizer,scheduler,flag='train')
            fit(epoch,model,devloader,optimizer,scheduler,flag='dev')


            