
import sys
sys.path.append('xxx')
import torch
import numpy as np
import os 
import pickle
import pandas as pd
from collections import deque,Counter
from scipy import stats
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
import re
from tqdm import tqdm
from nltk.corpus import stopwords
import random
from datetime import datetime
from collections import defaultdict
import json
import inflect

import json

SEED = 2019
torch.manual_seed(SEED)

p = inflect.engine()

class PatientDataset(object):
    def __init__(self, data_dir, Rationale_ts = False, inlcude_rationale = False,flag="train",):
        self.data_dir = data_dir
        self.flag = flag
        self.inlcude_rationale = inlcude_rationale
        self.text_dir = 'xxxx/dataset/brief_course/'
        # if Rationale_ts:

        #     self.new_text_dir = "xxxx/dataset/mimic_iii_rationale_with_ts/"
        # else:
        #     self.new_text_dir = "xxxx/dataset/mimic_iii_rationale/"
        self.new_text_dir = "xxxx/dataset/mimiciii_ratioanle_ts_split"
        print(f"Rationale dir: {self.new_text_dir }")

        self.numeric_dir = 'xxxx/dataset/alldata/all/'
 
        self.stopword = list(pd.read_csv('xxxx/stopwods.csv').values.squeeze())

        self.low = [2.80000000e+01, -7.50000000e-02,  4.30000000e+01, 4.00000000e+01,
                    4.10000000e+01,  9.00000000e+01,  5.50000000e+00,  6.15000000e+01,  
                    3.50000000e+01,  3.12996266e+01, 7.14500000e+00] 
        self.up = [  92.,           0.685,         187.,         128.,   
                    113.,         106.,          33.5,        177.5,         
                    38.55555556, 127.94021917,   7.585]   
        self.interpolation = [  59.0,           0.21,         128.0,         86.0,   
            77.0,         98.0,          19.0,        118.0,         
            36.6, 81.0,   7.4]
        self.patient_list = os.listdir(os.path.join(f'{data_dir}',flag))        
        self.max_length = 1000
        self.count = 0
        self.feature_list = [
        'Diastolic blood pressure',
        'Fraction inspired oxygen', 
        'Glucose', 
        'Heart Rate', 
        'Mean blood pressure', 
        'Oxygen saturation', 
        'Respiratory rate',
        'Systolic blood pressure', 
        'Temperature', 
        'Weight', 
        'pH']
        self.label_list = ["Acute and unspecified renal failure",
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
       

    def data_processing(self,data):

        return ''.join([i.lower() for i in data if not i.isdigit()])
   
    def sort_key(self,text):
        temp = []
        id_ = int(re.split(r'(\d+)', text.split("_")[-1])[1])
        temp.append(id_)
        return temp
    def rm_stop_words(self,text):
            tmp = text.split(" ")
            for t in self.stopword:
                while True:
                    if t in tmp:
                        tmp.remove(t)
                    else:
                        break
            text = ' '.join(tmp)
            # print(len(text))
            return text
    def __getitem__(self, idx):
        patient_file = self.patient_list[idx]
        text_df = pd.read_csv(self.text_dir+"_".join(patient_file.split("_")[:2])+".csv").values
        breif_course = text_df[:,1:2].tolist()
        breif_course = [str(i[0]) for i in breif_course if not str(i[0]).isdigit()]
        text = ' '.join(breif_course)
        text = self.rm_stop_words(text)
        rationale_file = json.load(open(os.path.join(self.new_text_dir,self.flag,patient_file.split(".csv")[0]+".json")))
        rationale = rationale_file['rationale']
        ts_rationale = rationale_file['lab_rationale']
        numeric_data_file =  self.numeric_dir + patient_file.split("_")[0] + "_" + patient_file.split("_")[2].replace("eposide","episode").strip(".csv") + "_timeseries.csv"
        lab_dic = defaultdict(list)

        lab_description = []

        if not os.path.exists(numeric_data_file):
            numeric_data = np.array([self.interpolation]*24)

            for n in self.feature_list:
                lab_description.append(f"{n.lower()} is normal all the time")


        else:
            self.count += 1
            numeric_data = pd.read_csv(numeric_data_file)[self.feature_list].values
            for l in range(numeric_data.shape[-1]):
                for s in np.array(numeric_data[:,l]):
                    if s <= self.low[l]:
                        lab_dic[l].append("low")
                    elif s > self.up[l]:
                        lab_dic[l].append("high")
                    else:
                        lab_dic[l].append("normal")
            # print()
            for k in lab_dic.keys():
                risk_types = set(lab_dic[k])
                if len(risk_types) ==1 and "normal" in risk_types:
                    descp = self.feature_list[k] + f" is normal all the time"
                else:
                    if "high" in risk_types:
                        number = p.number_to_words(len(np.where(np.array(lab_dic[k])== "high")[0]))
                        descp = self.feature_list[k] + f" is higher than normal {number} times"
                    if "low"  in risk_types:
                        number = p.number_to_words(len(np.where(np.array(lab_dic[k])== "low")[0]))
                        descp = self.feature_list[k] + f" is lower than normal {number} times"

                    if "high" in risk_types and  "low"  in risk_types:
                        high_number = p.number_to_words(len(np.where(np.array(lab_dic[k])== "high")[0]))
                        low_number = p.number_to_words(len(np.where(np.array(lab_dic[k])== "low")[0]))
                        descp = self.feature_list[k] + f" is higher than normal {high_number} times and is lower than normal {low_number} times"

                lab_description.append(descp.lower())

        lab_description = ','.join(lab_description)
 
        label = pd.read_csv(os.path.join(self.data_dir,self.flag,patient_file))[self.label_list].values[:1,:][0]
        label_index = np.where(label == 1)[0]
        label_names = [self.label_list[i] for i in label_index]
        if not label_names: label_names = ["no specific disease was identified"]
        label_names = ','.join(label_names)

        if len(numeric_data) < self.max_length:
            numeric_data = np.concatenate((numeric_data, np.repeat(np.expand_dims(numeric_data[-1,:], axis=0),1000-len(numeric_data),axis = 0) ), axis=0)
        return numeric_data,label_names,text,label,lab_description,rationale,ts_rationale,patient_file


    def __len__(self):
        return len(self.patient_list)

