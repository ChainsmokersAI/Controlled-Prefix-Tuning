import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_pretrained(model_name):
    """
    """
    print('Loading Pre-Trained Tokenizer, LM..')

    tokenizer=AutoTokenizer.from_pretrained(model_name)

    pretrained=AutoModelForCausalLM.from_pretrained(model_name)

    # Freeze LM
    for param in pretrained.parameters():
        param.requires_grad=False

    print('Loaded!\n')

    print('pad_token:', tokenizer.pad_token)
    print('pad_token_id:', tokenizer.pad_token_id, '\n')

    return tokenizer, pretrained

class AttrAlgnAC(nn.Module):
    """
    """
    def __init__(self, base_config, hidden_dim=512):
        super().__init__()

        # Config of Base (Pre-Trained) LM
        self.base_config=base_config

        # (Corpus Domain) Alignment Function
        self.algn_func_domain=nn.Sequential(
            nn.Linear(2*base_config.n_layer*base_config.n_embd,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,2*base_config.n_layer*base_config.n_embd)
        )

        # (Attribute) Alignment Function
        self.algn_func_attr=nn.Sequential(
            nn.Linear(2*base_config.n_layer*base_config.n_embd,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,2*base_config.n_layer*base_config.n_embd)
        )

    def transform(self, input_key_values, b_sz, seq_len):
        """
        Transform: Past_Key_Values -> Torch.Tensor # Shape (batch_size, seq_len, 2*n_layer*n_embd)
        """
        hiddens=[]
        for key, value in input_key_values:
            hiddens.append(key)
            hiddens.append(value)
            
        # 2*n_layer, batch_size, n_head, seq_len, n_embd/n_head
        hiddens=torch.stack(hiddens)
        # batch_size, seq_len, 2*n_layer, n_head, n_embd/n_head
        hiddens=hiddens.permute(1,3,0,2,4)
        # batch_size, seq_len, 2*n_layer*n_embd
        hiddens=hiddens.reshape(b_sz,seq_len,2*self.base_config.n_layer*self.base_config.n_embd)
        
        return hiddens
    
    def forward(self, domain_key_values, attr_key_values):
        # Batch Size
        b_sz=domain_key_values[0][0].shape[0]
        
        # Sequence Length
        seq_len_domain=domain_key_values[0][0].shape[2]
        seq_len_attr=attr_key_values[0][0].shape[2]
        
        # batch_size, seq_len, 2*n_layer*n_embd
        hiddens_domain=self.transform(input_key_values=domain_key_values, b_sz=b_sz, seq_len=seq_len_domain)
        hiddens_attr=self.transform(input_key_values=attr_key_values, b_sz=b_sz, seq_len=seq_len_attr)

        hiddens_domain=self.algn_func_domain(hiddens_domain)
        hiddens_attr=self.algn_func_attr(hiddens_attr)
        
        # batch_size, seq_len, 2*n_layer, n_head, n_embd/n_head
        hiddens_domain=hiddens_domain.reshape(b_sz,seq_len_domain,2*self.base_config.n_layer,self.base_config.n_head,int(self.base_config.n_embd/self.base_config.n_head))
        hiddens_attr=hiddens_attr.reshape(b_sz,seq_len_attr,2*self.base_config.n_layer,self.base_config.n_head,int(self.base_config.n_embd/self.base_config.n_head))
        
        # batch_size, seq_len(domain+attribute), 2*n_layer, n_head, n_embd/n_head
        hiddens=torch.cat([hiddens_domain, hiddens_attr], dim=1)
        
        # 2*n_layer, batch_size, n_head, seq_len, n_embd/n_head
        hiddens=hiddens.permute(2,0,3,1,4)
        
        # ALIGNED Past_Key_Values
        output_key_values=[(hidden[0], hidden[1]) for hidden in hiddens.chunk(self.base_config.n_layer)]
        
        return output_key_values

class AttrAlgnA(nn.Module):
    """
    """
    def __init__(self, base_config, hidden_dim=512):
        super().__init__()

        # Config of Base (Pre-Trained) LM
        self.base_config=base_config

        # Alignment Function
        self.algn_func=nn.Sequential(
            nn.Linear(2*base_config.n_layer*base_config.n_embd,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,2*base_config.n_layer*base_config.n_embd)
        )

    def forward(self, input_key_values):
        # Batch Size
        b_sz=input_key_values[0][0].shape[0]
        # Sequence Length
        seq_len=input_key_values[0][0].shape[2]
        
        hiddens=[]
        for key, value in input_key_values:
            hiddens.append(key)
            hiddens.append(value)
            
        # 2*n_layer, batch_size, n_head, seq_len, n_embd/n_head
        hiddens=torch.stack(hiddens)
        # batch_size, seq_len, 2*n_layer, n_head, n_embd/n_head
        hiddens=hiddens.permute(1,3,0,2,4)
        # batch_size, seq_len, 2*n_layer*n_embd
        hiddens=hiddens.reshape(b_sz,seq_len,2*self.base_config.n_layer*self.base_config.n_embd)
        
        hiddens=self.algn_func(hiddens)
        
        # batch_size, seq_len, 2*n_layer, n_head, n_embd/n_head
        hiddens=hiddens.reshape(b_sz,seq_len,2*self.base_config.n_layer,self.base_config.n_head,int(self.base_config.n_embd/self.base_config.n_head))
        # 2*n_layer, batch_size, n_head, seq_len, n_embd/n_head
        hiddens=hiddens.permute(2,0,3,1,4)
        
        output_key_values=[(hidden[0], hidden[1]) for hidden in hiddens.chunk(self.base_config.n_layer)]
        
        return output_key_values

class ControlPrefixes(nn.Module):
    """
    """
    def __init__(self, base_config, control_config, preseqlen=5, hidden_dim=512):
        super().__init__()

        # Config of Base (Pre-Trained) LM
        self.base_config=base_config
        # Config of Control-Codes
        control_config['pad']=0
        self.control_config=control_config

        # Input: 0, 1, 2 ... preseqlen (General-Prefix)
        self.preseq=torch.arange(preseqlen)
        # Embedding for General-Prefix
        self.embd_general=nn.Embedding(preseqlen,base_config.n_embd)
        # Embedding for Control-Prefix
        self.embd_control=nn.Embedding(len(control_config),base_config.n_embd)
        # Reparam (Shared Between General-Prefix & Control-Prefix)
        self.reparam=nn.Sequential(
            nn.Linear(base_config.n_embd,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,2*base_config.n_layer*base_config.n_embd)
        )

    def forward(self, batch_size, control, device):
        # General-Prefix: batch_size, preseqlen
        preseq=self.preseq.unsqueeze(0).expand(batch_size,-1).to(device)
        # General-Prefix: batch_size, preseqlen, n_embd
        preseq=self.embd_general(preseq)

        # Control-Prefix
        preseq_control=[[self.control_config[code] for code in codes] for codes in control]
        # Control-Prefix: batch_size, len(control_codes)
        preseq_control=torch.tensor(preseq_control).to(device)
        # Control-Prefix: batch_size, len(control_codes), n_embd
        preseq_control=self.embd_control(preseq_control)

        # Merge: [Control-Prefix, General-Prefix]
        # batch_size, len(control_codes)+preseqlen, n_embd
        preseq=torch.cat((preseq_control,preseq), dim=1)
        # batch_size, len(control_codes)+preseqlen, 2*n_layer*n_embd
        preseq=self.reparam(preseq)
        # batch_size, len(control_codes)+preseqlen, 2*n_layer, n_head, n_embd/n_head
        preseq=preseq.reshape(batch_size,len(control[0])+len(self.preseq),2*self.base_config.n_layer,self.base_config.n_head,int(self.base_config.n_embd/self.base_config.n_head))
        # 2*n_layer, batch_size, n_head, len(control_codes)+preseqlen, n_embd/n_head
        past_key_values=preseq.permute(2,0,3,1,4)

        return past_key_values.split(2)

class PrefixTuning(nn.Module):
    """
    """
    def __init__(self, base_config, preseqlen=5, hidden_dim=512):
        super().__init__()

        # Config of Base (Pre-Trained) LM
        self.base_config=base_config

        # Input: 0, 1, 2 ... preseqlen
        self.preseq=torch.arange(preseqlen)
        # Embedding
        self.embd=nn.Embedding(preseqlen,base_config.n_embd)
        # Reparam
        self.reparam=nn.Sequential(
            nn.Linear(base_config.n_embd,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,2*base_config.n_layer*base_config.n_embd)
        )

    def forward(self, batch_size, device):
        # batch_size, preseqlen
        preseq=self.preseq.unsqueeze(0).expand(batch_size,-1).to(device)
        # batch_size, preseqlen, n_embd
        preseq=self.embd(preseq)
        # batch_size, preseqlen, 2*n_layer*n_embd
        preseq=self.reparam(preseq)
        # batch_size, preseqlen, 2*n_layer, n_head, n_embd/n_head
        preseq=preseq.reshape(batch_size,len(self.preseq),2*self.base_config.n_layer,self.base_config.n_head,int(self.base_config.n_embd/self.base_config.n_head))
        # 2*n_layer, batch_size, n_head, preseqlen, n_embd/n_head
        past_key_values=preseq.permute(2,0,3,1,4)

        return past_key_values.split(2)
