from os import sep
from pyexpat.errors import codes
import pandas as pd

import torch
from torch.utils.data import Dataset

def load_config_for_control_prefixes(path_data):
    """
    Dict of Control-Codes in Dataset.
    Need for Training Control-Prefixes Model.
    e.g.
    {
        ' positive ': 1,
        ' negative ': 2
    }
    or
    {
        ' economy ': 1,
        ' society ': 2,
        ' sports ': 3,
        ...
    }
    """
    dict_codes={}

    codes=pd.read_excel(path_data)['code']
    for code in codes:
        for c in code.split('|')[1:]:
            if c not in dict_codes: dict_codes[c]=len(dict_codes)+1

    return dict_codes

class DatasetAttrAlgn(Dataset):
    """
    """
    def __init__(self, path_data, tokenizer):
        self.data=[]
        self.label=[]
        # Attribute(s)
        self.attr=[]
        
        print('Processing Data..')
        
        # Load Data
        df_train=pd.read_excel(path_data)
        for index in df_train.index:
            code=df_train.loc[index]['code']
            text=df_train.loc[index]['record']
            
            # Debug
            if type(code)!=str or type(text)!=str: continue

            # Data: <BOS> Text <EOS>
            data=tokenizer.encode(tokenizer.bos_token+text+tokenizer.eos_token)
            self.data.append(data)

            # Label: <BOS> Text <EOS>
            # Same as Data on Default Setting
            # May Be Different with Data on Customized Setting
            label=tokenizer.encode(tokenizer.bos_token+text+tokenizer.eos_token)
            self.label.append(label)

            # Attribute(s): '| economy | Sports | ..'
            attr=tokenizer.encode(code)
            self.attr.append(attr)

        print(len(self.data), 'Data Processed!\n')
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.attr[idx]
    
    def __len__(self):
        return len(self.data)

class DatasetControlPrefixes(Dataset):
    """
    """
    def __init__(self, path_data, tokenizer):
        self.data=[]
        self.label=[]
        # Control-Code (Class)
        self.control=[]
        
        print('Processing Data..')
        
        # Load Data
        df_train=pd.read_excel(path_data)
        for index in df_train.index:
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

            # Control-Code: '| economy | sports ':str -> [' economy ', ' sports ']:list
            self.control.append(code.split('|')[1:])

        print(len(self.data), 'Data Processed!\n')
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.control[idx]
    
    def __len__(self):
        return len(self.data)

class DatasetPrefixTuning(Dataset):
    """
    """
    def __init__(self, path_data, tokenizer):
        self.data=[]
        self.label=[]
        
        print('Processing Data..')
        
        # Load Data
        df_train=pd.read_excel(path_data)
        for index in df_train.index:
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

def collate_fn_attr_algn(pad_token_id, domain=None):
    """
    """
    def collate_fn(batch):
        max_len=0
        max_len_attr=0
        for data, _, attr in batch:
            if len(data)>max_len: max_len=len(data)
            if len(attr)>max_len_attr: max_len_attr=len(attr)
                
        datas=[]
        labels=[]
        attrs=[]
        for data, label, attr in batch:
            data.extend([pad_token_id]*(max_len-len(data)))
            datas.append(data)
            
            label.extend([pad_token_id]*(max_len-len(label)))
            labels.append(label)

            attr.extend([pad_token_id]*(max_len_attr-len(attr)))
            attrs.append(attr)

        # for Method: A (No Corpus Domain Disentanglement)
        if domain==None:
            return torch.tensor(datas), torch.tensor(labels), torch.tensor(attrs), None
        # for Method: AC
        else:
            return torch.tensor(datas), torch.tensor(labels), torch.tensor(attrs), torch.tensor(len(batch)*[domain])
            
    return collate_fn

def collate_fn_control_prefixes(pad_token_id):
    """
    """
    def collate_fn(batch):
        max_len=0
        max_len_control=0
        for data, _, control in batch:
            if len(data)>max_len: max_len=len(data)
            if len(control)>max_len_control: max_len_control=len(control)
                
        datas=[]
        labels=[]
        controls=[]
        for data, label, control in batch:
            data.extend([pad_token_id]*(max_len-len(data)))
            datas.append(data)
            
            label.extend([pad_token_id]*(max_len-len(label)))
            labels.append(label)

            control.extend(['pad']*(max_len_control-len(control)))
            controls.append(control)
            
        return torch.tensor(datas), torch.tensor(labels), controls

    return collate_fn

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

