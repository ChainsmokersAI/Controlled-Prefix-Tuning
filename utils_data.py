from os import sep
import pandas as pd

import torch
from torch.utils.data import Dataset

class DatasetPrefixTuning(Dataset):
    """
    """
    def __init__(self, path_data, tokenizer):
        self.data=[]
        self.label=[]
        
        print('Processing Data..')
        
        # Load Data
        df_train=pd.read_excel(path_data)
        for index in df_train.index[:100]:
            code=df_train.loc[index]['code']
            text=df_train.loc[index]['record']

            # Debug
            if type(code)!=str or type(text)!=str: continue

            # Data: Control-Code <BOS> Text <EOS>
            data=tokenizer.encode(code+tokenizer.bos_token+text+tokenizer.eos_token)
            self.data.append(data)

            # Label: -100 -100 ... -100 Text <EOS>
            label=tokenizer.encode(code+tokenizer.bos_token+text+tokenizer.eos_token)
            sep=data.index(tokenizer.bos_token_id)+1
            # Masking
            label[:sep]=[-100]*sep
            self.label.append(label)

        print(len(self.data), 'Data Processed!\n')
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def __len__(self):
        return len(self.data)

def collate_fn_prefix_tuning(pad_token_id):
    """
    """
    def collate_fn(batch):
        max_len=0
        for data, _ in batch:
            if len(data)>max_len: max_len=len(data)
                
        datas=[]
        labels=[]
        for data, label in batch:
            data.extend([pad_token_id]*(max_len-len(data)))
            datas.append(data)
            
            label.extend([pad_token_id]*(max_len-len(label)))
            labels.append(label)
            
        return torch.tensor(datas), torch.tensor(labels)

    return collate_fn
