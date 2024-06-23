
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
from models.model_clinragen import MEDTS
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
SEED = xxx #gpu23 model 2
import json

import evaluate
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch", default=12, type = int)
parser.add_argument("--rt", action= "store_true")
parser.add_argument("--kg", action= "store_true")
parser.add_argument("--rationale_ts", action= "store_true")
parser.add_argument("--eval", action= "store_true")
parser.add_argument("--sv_weight", action= "store_true")
parser.add_argument("--logs", action= "store_true")
args = parser.parse_args()  
args = parser.parse_args()  


torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
# conda activate qformer
model_name = "google/flan-t5-small"
num_epochs = 1000
max_length = 1600
BATCH_SIZE = args.batch
patch_len = 8
num_patch = 125
prompt_length = 24
pretrained = True
SV_WEIGHTS = args.sv_weight
logs = args.logs
evaluation = args.eval
Rationale_ts = args.rt
knwoledge_asinput = args.kg
inlcude_rationale = True
Freeze_t5coder = False
Freeze_TST = True
rationale_ts = args.rationale_ts
Best_F1 = 0.62
date = str(datetime.date.today())
save_dir= "xxxx/llm_rationale_medts/weights/mimiciv"
sv_model_id = model_name.split("/")[-1]
gpuid = socket.gethostname()
save_name = f"{sv_model_id}_{date}_medts_disease_rationalets_{args.rt}_{SEED}_{gpuid}_fzt5-{Freeze_t5coder}_rationale_ts-{rationale_ts}"
log_file_name = f'xxxx/llm_rationale_medts/logs/{save_name}.txt'
t5_encoder_weight_dir = "xx.pth"
ts_encoder_weight_dir = "xx.pth"


start_epoch = 0
print("#####################")
print(f"Server: {gpuid}, date: {date}, SEED: {SEED}, model_name: {model_name}, pretrained: {pretrained}, BATCH_SIZE: {BATCH_SIZE}, knwoledge_asinput:{knwoledge_asinput}, Rationale_ts: {Rationale_ts}, inlcude_rationale: {inlcude_rationale}, max_length: {max_length}, Freeze_t5coder: {Freeze_t5coder},  "+'\n')
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
    weight_dir = "xx.pth"
else:
    with open(f'{log_file_name}', 'a+') as log_file:
        log_file.write(f"Server: {gpuid}, date: {date}, SEED: {SEED}, model_name: {model_name}, pretrained: {pretrained}, BATCH_SIZE: {BATCH_SIZE}, knwoledge_asinput:{knwoledge_asinput}, Rationale_ts: {Rationale_ts}, inlcude_rationale: {inlcude_rationale}, max_length: {max_length}, Freeze_TST: {Freeze_TST} "+'\n')
        log_file.close()

tokenizer = AutoTokenizer.from_pretrained(model_name)

accelerator = Accelerator(
    mixed_precision="no",
    device_placement=True,
)
device = accelerator.device

# def collate_fn(data):

#     y = [i[0] for i in data]
#     text = [i[1] for i in data]
#     # text = [f"Diagnose disease from the following medical notes:\n{d[1]}\n Diseases: \n" for d in data]
#     label_list = [i[2] for i in data]

#     return y,text,label_list


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [sq[0].shape[0] for sq in data]
    input_x = [i[0].tolist() for i in data]
    disease = [f"### Disease Diagnosed: {i[1]}" for i in data]
    input_x = rnn_utils.pad_sequence([torch.from_numpy(np.array(x)) for x in input_x],batch_first = True, padding_value=0)

    text = [f"Diagnose disease from the following medical notes: \n{d[2]}\n and Lab Test: \n" for d in data]
    label_list = [i[3] for i in data]
    lab_description = [i[4] for i in data]

    disease_with_rationale = [f"### Disease Diagnosed: {d[1]}. ### Rationale: {d[5]}" for d in data]
    if Rationale_ts:
        disease_with_rationale = [f"### Disease Diagnosed: {d[1]}. ### Rationale: {d[5]} {d[6]}" for d in data]

    return input_x,disease_with_rationale,text,label_list,lab_description



def fit(epoch,model,dataloader,optimizer,scheduler,flag='train'):
    global Best_F1,similar_model
    if flag == 'train':
        model.train()
    else:
        model.eval()
    batch_loss_list = []
    y_list = []
    pred_list_f1 = []
    bleu_list = []
    bert_score_list = []
    model = model.to(device)
    similar_model.eval()
    similar_model = similar_model.to(device)
    embedding_labels = similar_model.encode(target_diagnosis_name_list, convert_to_tensor=True)

    for i,(lab_x,labels,text_list,label_list,lab_description) in enumerate(tqdm(dataloader)):
        label = torch.tensor(np.array(label_list)).to(torch.float32).to(device)
        # if i == 10:break
        if flag == "train":
            with torch.set_grad_enabled(True):
                lab_x = torch.tensor(lab_x).to(torch.float32).to(device)
                lab_x = lab_x.view(lab_x.shape[0],num_patch,lab_x.shape[-1],patch_len)
                text_input = tokenizer(text_list, truncation = True, return_tensors="pt",pad_to_max_length=True, padding="max_length",max_length = max_length).to(device)
                labdescp_input = tokenizer(lab_description, return_tensors="pt",padding=True).to(device)
                label_input = tokenizer(labels, return_tensors="pt",padding=True).to(device)
                loss,mm_input =  model(lab_x,label_input,text_input,labdescp_input)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                lab_x = torch.tensor(lab_x).to(torch.float32).to(device)
                lab_x = lab_x.view(lab_x.shape[0],num_patch,lab_x.shape[-1],patch_len)
                text_input = tokenizer(text_list, truncation = True, return_tensors="pt",pad_to_max_length=True, padding="max_length",max_length = max_length).to(device)
                labdescp_input = tokenizer(lab_description, return_tensors="pt",padding=True).to(device)
                label_input = tokenizer(labels, return_tensors="pt",padding=True).to(device)
                # print(labels)

                loss,mm_input =  model(lab_x,label_input,text_input,labdescp_input)
                output_sequences = model.t5_decoder.generate(
                    inputs_embeds = mm_input,
                    # input_ids = text_input["input_ids"],
                    num_beams = 2,
                    max_length = 500,
                    temperature = 0.8,
                    num_return_sequences = 1,
                )  


                pred_labels = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                # print(pred_labels)
                pred = []
                for n, pred_label_ in enumerate(pred_labels):
                    s_pred = [0]*embedding_labels.shape[0]
                    if "### Disease Diagnosed: " in pred_label_:
                        pred_label = pred_label_.split("### Disease Diagnosed: ")[1]
                        if ". ### Rationale" in pred_label:
                            pred_label = pred_label.split(". ### Rationale: ")[0]
                    else:
                        pred_label = pred_label_
                    embedding_preds = similar_model.encode(pred_label.split(","), convert_to_tensor=True)
                    for j in range(embedding_labels.shape[0]): 
                        embedding_1= embedding_labels[j,:]
                        for k in range(embedding_preds.shape[0]):
                            embedding_2 = embedding_preds[k,:]
                            if util.pytorch_cos_sim(embedding_1, embedding_2) >= 0.9:
                                s_pred[j] = 1  
                    pred.append(s_pred)
                    if ". ### Rationale" in pred_label_:


                        pred_rationale = pred_label_.split(". ### Rationale: ")[1]

                        label_rationale =  labels[n].split(". ### Rationale: ")[1]

                        bleu_result = bleu.compute(predictions=[pred_rationale], references=[label_rationale])["bleu"]
                        # bleu_result = 0
                        bleu_list.append(bleu_result)

                        bert_score_results = bertscore.compute(predictions=[pred_rationale], references=[label_rationale], lang="en")["f1"][0]
                        # bert_score_results = 0
                        bert_score_list.append(bert_score_results)

                pred = np.array(pred)   
                y = np.array(label.cpu().data.tolist())

                # print("..............................")
                y_list.append(y)
                pred_list_f1.append(pred)
                batch_loss_list.append( loss.cpu().data )  
    accelerator.wait_for_everyone()
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
        if not bleu_list: bleu_list = [0]
        if not bert_score_list: bert_score_list = [0]
        bleu_score = sum(bleu_list)/ len(bleu_list)
        bert_score = sum(bert_score_list)/ len(bert_score_list)
        total_loss = sum(batch_loss_list) / len(batch_loss_list)
        end = time.time()
        running_time = end - start
        print("PHASE: {} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  ACC: {} | BLEU: {} | Bert Score: {} | Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,acc,bleu_score,bert_score,total_loss))
    
        if SV_WEIGHTS:
            if logs:
                with open(f'{log_file_name}', 'a+') as log_file:
                    log_file.write("PHASE: {} EPOCH : {} | Running time: {} |  Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  ACC: {} | Total LOSS  : {}  ".format(flag,epoch + 1, running_time, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,acc,total_loss)+'\n')
                    log_file.close()
            if f1_micro > Best_F1:
                Best_F1 = f1_micro
                PATH=f"{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(loss),4)}_f1_{round(float(f1_micro),4)}_acc_{round(float(acc),4)}.pth"
                best_model_wts = copy.deepcopy(accelerator.get_state_dict(model))
                torch.save(best_model_wts, PATH)

        



if __name__ == '__main__':

    train_dataset = PatientDataset('mimicdata/mimic4/', flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    dev_dataset = PatientDataset('mimicdata/mimic4/', flag="dev")
    devloader = torch.utils.data.DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)


    print(train_dataset.__len__())
    print(dev_dataset.__len__())
    model = MEDTS(model_name, prompt_length, knwoledge_asinput = knwoledge_asinput, Freeze_t5coder = Freeze_t5coder, Freeze_TST = Freeze_TST)

    if pretrained:
        if evaluation:
            model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device)), strict=False)
            print("loading validation weight: ",weight_dir)
        else:
            model.load_state_dict(torch.load(ts_encoder_weight_dir,map_location=torch.device(device)), strict=False)
            print("loading weight: ",ts_encoder_weight_dir)
            model.load_state_dict(torch.load(t5_encoder_weight_dir,map_location=torch.device(device)), strict=False)
            print("loading weight: ",t5_encoder_weight_dir)
    optimizer = AdamW(model.parameters(True), lr=2e-5, eps = 1e-8, weight_decay = 0.05)

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


