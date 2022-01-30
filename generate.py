import argparse
import sys
import types

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

parser=argparse.ArgumentParser(description='Controlled Generation using Prompt Learning (Trained Model)')
parser.add_argument('--model', type=str, required=True, help='Model: attr-algn | control-prefixes | prefix-tuning')
parser.add_argument('--path', type=str, required=True, help='Path of Tained Model')
parser.add_argument('--device', type=str, default='gpu', help='Device where Model Trained on: gpu(default) | cpu')
parser.add_argument('--code', type=str, required=True, help='Control Codes: | economy | sports | ..')
parser.add_argument('--prompt', type=str, default='', help='Prompt Text: Last night US..')
parser.add_argument('--n_sample', type=int, default=5, help='Number of Sampled Generations: 5(default)')
parser.add_argument('--max_len', type=int, default=100, help='Max Length of Generated Texts: 100(default)')
parser.add_argument('--p', type=float, default=0.8, help='P of Nucleus(Top-p) Sampling')
parser.add_argument('--method', type=str, default='A', help='Method of Attribute-Alignment: A(default) | AC')
parser.add_argument('--domain', type=str, default='domain', help='Corpus Domain for Attribute-Alignment (AC)')
#parser.add_argument('', type=, required=, help=)
#parser.add_argument('', type=, default=, help=)
args=parser.parse_args()

def get_prefix(tokenizer, pretrained, model, device):
    """
    """
    if args.model=='prefix-tuning':
        prefix=model(batch_size=1, device=device)
    elif args.model=='control-prefixes':
        control=[args.code.split('|')[1:]]
        prefix=model(batch_size=1, control=control, device=device)
    elif args.model=='attr-algn':
        if args.method=='A':
            attr=tokenizer.encode(args.code, return_tensors='pt').to(device)

            encoded_attr=pretrained(input_ids=attr, use_cache=True)['past_key_values']

            prefix=model(input_key_values=encoded_attr)
        elif args.method=='AC':
            attr=tokenizer.encode(args.code, return_tensors='pt').to(device)
            domain=tokenizer.encode(args.domain, return_tensors='pt').to(device)

            encoded_attr=pretrained(input_ids=attr, use_cache=True)['past_key_values']
            encoded_domain=pretrained(input_ids=domain, use_cache=True)['past_key_values']

            prefix=model(domain_key_values=encoded_domain, attr_key_values=encoded_attr)

    return prefix

def load_model(device):
    """
    """
    # Load Trained Model
    model=torch.load(args.path).to(device)
    # Config of Base LM
    base_config=model.base_config
    print('Base LM:', base_config._name_or_path, '\n')
    
    # Load Base (Pre-Trained) Tokenizer, LM
    tokenizer=AutoTokenizer.from_pretrained(base_config._name_or_path)
    pretrained=AutoModelForCausalLM.from_pretrained(base_config._name_or_path).to(device)

    if tokenizer.pad_token==None:
        # Add PAD Token: [PAD]
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        pretrained.resize_token_embeddings(len(tokenizer))

    # Bind Customized Generation Function To Base LM
    sys.path.append('./transformers/')
    from customized_generation_utils import generate, sample
    
    pretrained.generate=types.MethodType(generate, pretrained)
    pretrained.sample=types.MethodType(sample, pretrained)

    return tokenizer, pretrained, model

def do_generate(device):
    """
    Generation Strategy: Nucleus Sampling
    """
    print('\n***** Controlled Generation *****')
    print('Model:', args.model)
    if args.model=='attr-algn':
        print('Method:', args.method)
    if args.model=='attr-algn' and args.method!='A':
        print('Domain:', args.domain)
    print('Path:', args.path)
    print('Control Codes:', args.code)
    print('Prompt:', args.prompt)
    print('Device:', device)
    print('***********************************\n')

    # Load Base LM & Trained Model
    tokenizer, pretrained, model=load_model(device=device)

    # Get Past_Key_Values
    prefix=get_prefix(tokenizer=tokenizer, pretrained=pretrained, model=model, device=device)

    # Input Text
    if args.model=='attr-algn':
        # <BOS> Prompt
        input_=tokenizer.bos_token+args.prompt
    else:
        # Control-Codes <BOS> Prompt
        input_=args.code+tokenizer.bos_token+args.prompt

    # Repeat Sampling for 'n_sample' Times
    for n in range(args.n_sample):
        # Nucleus Sampling
        outputs=pretrained.generate(
            tokenizer.encode(input_, return_tensors='pt').to(device),
            do_sample=True,
            max_length=args.max_len,
            top_p=args.p,
            top_k=0,
            prefix=prefix
        )
        # Decode
        generated=tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("-----")
        print(generated)
    print('\n')

def main():
    if args.model not in ['prefix-tuning', 'control-prefixes', 'attr-algn']:
        print('Wrong Model Name!')
        return

    if args.device=='gpu' and torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')

    do_generate(device=device)

if __name__=='__main__':
    main()
