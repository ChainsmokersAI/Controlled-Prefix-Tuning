import code
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

