import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from transformers import AdamW, get_linear_schedule_with_warmup

from utils_data import DatasetAttrAlgn, DatasetControlPrefixes, DatasetPrefixTuning, collate_fn_attr_algn, collate_fn_control_prefixes, collate_fn_prefix_tuning, load_config_for_control_prefixes
from models import load_pretrained, AttrAlgnAC, AttrAlgnA, ControlPrefixes, PrefixTuning

parser=argparse.ArgumentParser(description='Controlled Generation using Prompt Learning')
parser.add_argument('--dataset', type=str, required=True, help='Dataset Path')
parser.add_argument('--model', type=str, required=True, help='Model: attr-algn | control-prefixes | prefix-tuning')
parser.add_argument('--base', type=str, required=True, help='Base (Pre-Trained) LM: gpt2-large etc')
parser.add_argument('--device', type=str, default='gpu', help='Device where Model Trained on: gpu(default) | cpu')
parser.add_argument('--ddp', type=str, default='False', help='Multi-GPU Setting: True | False(default)')
parser.add_argument('--batch', type=int, default=4, help='Batch Size')
parser.add_argument('--accum', type=int, default=32, help='Accumulation Steps')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning Rate')
parser.add_argument('--epoch', type=int, default=5, help='Epochs')
parser.add_argument('--hidden', type=int, default=512, help='Hidden Dimension Size')
parser.add_argument('--preseqlen', type=int, default=5, help='Prefix Sequence Length for Prefix-Tuning, Control-Prefixes: 5(default)')
parser.add_argument('--method', type=str, default='A', help='Method of Attribute-Alignment: A(default) | AC')
parser.add_argument('--domain', type=str, default='domain', help='Corpus Domain for Attribute-Alignment (AC)')
#parser.add_argument('', type=, default=, help=)
args=parser.parse_args()

def train_ddp_attr_algn(rank, world_size):
    """
    Paper: Attribute Alignment: Controlling Text Generation from Pre-trained Language Models (https://arxiv.org/abs/2103.11070)
    """
    if rank==0:
        print('\n***** Attribute-Alignment *****')
        print('Method:', args.method)
        if args.method!='A':
            print('Corpus Domain:', args.domain)
        print('Batch Size:', args.batch*args.accum*world_size)
        print('Learning Rate:', args.lr)
        print('Epochs:', args.epoch)
        print('Hidden Dimension Size:', args.hidden)
        print('Number of GPUs:', world_size)
        print('******************************\n')

    # Create Default Process Group
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:8973', rank=rank, world_size=world_size)
    
    # Load Pre-Trained Tokenizer, LM
    tokenizer, pretrained=load_pretrained(args.base)
    pretrained=pretrained.to(rank)

    # Load Dataset
    dataset=DatasetAttrAlgn(path_data=args.dataset,tokenizer=tokenizer)
    sampler=DistributedSampler(dataset)
    # Collate Function: Padding for Same Sequence Length on Same Batch
    if args.method=='A':
        collate_fn=collate_fn_attr_algn(pad_token_id=tokenizer.pad_token_id)
    elif args.method=='AC':
        collate_fn=collate_fn_attr_algn(pad_token_id=tokenizer.pad_token_id, domain=tokenizer.encode(args.domain))
    # DataLoader
    dataloader=DataLoader(dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, sampler=sampler)
    
    # Load Model (on Device)
    if args.method=='A':
        model=AttrAlgnA(base_config=pretrained.config, hidden_dim=args.hidden)
    elif args.method=='AC':
        model=AttrAlgnAC(base_config=pretrained.config, hidden_dim=args.hidden)
    model=model.to(rank)
    model_ddp=DDP(model, device_ids=[rank])
        
    # Optim, Scheduler
    optimizer=AdamW(model_ddp.parameters(), lr=args.lr)
    scheduler=get_linear_schedule_with_warmup(
        optimizer=optimizer,
        # 3% of Total Steps
        num_warmup_steps=int(0.03*args.epoch*len(dataset)/(world_size*args.accum*args.batch)),
        num_training_steps=int(args.epoch*len(dataset)/(world_size*args.accum*args.batch))
    )

    # TensorBoard: Logging
    writer=SummaryWriter()
    step_global=0

    # Training
    for epoch in range(args.epoch):
        # Train Phase
        model_ddp.train()
        sampler.set_epoch(epoch)

        loss_train=0
        optimizer.zero_grad()

        for step, (data, label, attr, domain) in enumerate(dataloader):
            data=data.to(rank)
            label=label.to(rank)
            attr=attr.to(rank)

            # Attribute(s) Representation from Base (Pre-Trained) LM
            encoded_attr=pretrained(input_ids=attr, use_cache=True)['past_key_values']

            if args.method=='A':
                # Transform Attribute(s) Representation via Alignment Function
                past_key_values=model_ddp(input_key_values=encoded_attr)
            elif args.method=='AC':
                # Corpus Domain Representation from Base (Pre-Trained) LM
                encoded_domain=pretrained(input_ids=domain.to(rank), use_cache=True)['past_key_values']
                # Transform Attribute(s) Representation via Alignment Function
                past_key_values=model_ddp(domain_key_values=encoded_domain, attr_key_values=encoded_attr)

            # Forward: Base (Pre-Trained) LM
            outputs=pretrained(input_ids=data, labels=label, past_key_values=past_key_values)
            
            loss=outputs[0]/args.accum
            loss.backward()
            loss_train+=loss.item()
            
            if (step+1)%args.accum==0:
                step_global+=1
                
                if rank==0:
                    # TensorBoard
                    writer.add_scalar(
                        f'loss_train/Attribute-Alignment({args.method})_hidden{args.hidden}_batch{args.batch*args.accum*world_size}_lr{args.lr}_epoch{args.epoch}',
                        loss_train,
                        step_global
                    )

                # Set Loss to 0
                loss_train=0

                optimizer.step()
                scheduler.step()
                
                optimizer.zero_grad()
                
    if rank==0:
        # Save Model
        model_ddp.to(torch.device('cpu'))
        torch.save(model_ddp.module, f'./model/Attribute-Alignment({args.method})_DDP({world_size})_hidden{args.hidden}_batch{args.batch*args.accum*world_size}_lr{args.lr}_epoch{args.epoch}.pt')

def train_ddp_control_prefixes(rank, world_size):
    """
    Paper: Control Prefixes for Text Generation (https://arxiv.org/abs/2110.08329)
    """
    if rank==0:
        print('\n***** Control-Prefixes *****')
        print('Batch Size:', args.batch*args.accum*world_size)
        print('Learning Rate:', args.lr)
        print('Epochs:', args.epoch)
        print('Prefix Sequence Length:', args.preseqlen)
        print('Hidden Dimension Size:', args.hidden)
        print('Number of GPUs:', world_size)
        print('******************************\n')

    # Create Default Process Group
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:8973', rank=rank, world_size=world_size)
    
    # Load Pre-Trained Tokenizer, LM
    tokenizer, pretrained=load_pretrained(args.base)
    pretrained=pretrained.to(rank)

    # Load Dataset
    dataset=DatasetControlPrefixes(path_data=args.dataset,tokenizer=tokenizer)
    sampler=DistributedSampler(dataset)
    # Collate Function: Padding for Same Sequence Length on Same Batch
    collate_fn=collate_fn_control_prefixes(pad_token_id=tokenizer.pad_token_id)
    # DataLoader
    dataloader=DataLoader(dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, sampler=sampler)

    # Config of Control-Codes for Control-Prefixes
    control_config=load_config_for_control_prefixes(args.dataset)
    # Load Model (on Device)
    model=ControlPrefixes(base_config=pretrained.config, control_config=control_config, preseqlen=args.preseqlen, hidden_dim=args.hidden)
    model=model.to(rank)
    model_ddp=DDP(model, device_ids=[rank])
    
    # Optim, Scheduler
    optimizer=AdamW(model_ddp.parameters(), lr=args.lr)
    scheduler=get_linear_schedule_with_warmup(
        optimizer=optimizer,
        # 3% of Total Steps
        num_warmup_steps=int(0.03*args.epoch*len(dataset)/(world_size*args.accum*args.batch)),
        num_training_steps=int(args.epoch*len(dataset)/(world_size*args.accum*args.batch))
    )

    # TensorBoard: Logging
    writer=SummaryWriter()
    step_global=0

    # Training
    for epoch in range(args.epoch):
        # Train Phase
        model_ddp.train()
        sampler.set_epoch(epoch)

        loss_train=0
        optimizer.zero_grad()

        for step, (data, label, control) in enumerate(dataloader):
            data=data.to(rank)
            label=label.to(rank)
            
            # Get Past-Key-Values
            past_key_values=model_ddp(batch_size=data.shape[0], control=control, device=rank)
            # Forward: Base (Pre-Trained) LM
            outputs=pretrained(input_ids=data, labels=label, past_key_values=past_key_values)
            
            loss=outputs[0]/args.accum
            loss.backward()
            loss_train+=loss.item()
            
            if (step+1)%args.accum==0:
                step_global+=1
                
                if rank==0:
                    # TensorBoard
                    writer.add_scalar(
                        f'loss_train/Control-Prefixes_preseqlen{args.preseqlen}_hidden{args.hidden}_batch{args.batch*args.accum*world_size}_lr{args.lr}_epoch{args.epoch}',
                        loss_train,
                        step_global
                    )

                # Set Loss to 0
                loss_train=0

                optimizer.step()
                scheduler.step()
                
                optimizer.zero_grad()
                
    if rank==0:
        # Save Model
        model_ddp.to(torch.device('cpu'))
        torch.save(model_ddp.module, f'./model/Control-Prefixes_DDP({world_size})_preseqlen{args.preseqlen}_hidden{args.hidden}_batch{args.batch*args.accum*world_size}_lr{args.lr}_epoch{args.epoch}.pt')

def train_ddp_prefix_tuning(rank, world_size):
    """
    Paper: Prefix-Tuning: Optimizing Continuous Prompts for Generation (https://arxiv.org/abs/2101.00190)
    """
    if rank==0:
        print('\n***** Prefix-Tuning *****')
        print('Batch Size:', args.batch*args.accum*world_size)
        print('Learning Rate:', args.lr)
        print('Epochs:', args.epoch)
        print('Prefix Sequence Length:', args.preseqlen)
        print('Hidden Dimension Size:', args.hidden)
        print('Number of GPUs:', world_size)
        print('******************************\n')

    # Create Default Process Group
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:8973', rank=rank, world_size=world_size)
    
    # Load Pre-Trained Tokenizer, LM
    tokenizer, pretrained=load_pretrained(args.base)
    pretrained=pretrained.to(rank)

    # Load Dataset
    dataset=DatasetPrefixTuning(path_data=args.dataset,tokenizer=tokenizer)
    sampler=DistributedSampler(dataset)
    # Collate Function: Padding for Same Sequence Length on Same Batch
    collate_fn=collate_fn_prefix_tuning(pad_token_id=tokenizer.pad_token_id)
    # DataLoader
    dataloader=DataLoader(dataset, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, sampler=sampler)

    # Load Model (on Device)
    model=PrefixTuning(base_config=pretrained.config, preseqlen=args.preseqlen, hidden_dim=args.hidden)
    model=model.to(rank)
    model_ddp=DDP(model, device_ids=[rank])
    
    # Optim, Scheduler
    optimizer=AdamW(model_ddp.parameters(), lr=args.lr)
    scheduler=get_linear_schedule_with_warmup(
        optimizer=optimizer,
        # 3% of Total Steps
        num_warmup_steps=int(0.03*args.epoch*len(dataset)/(world_size*args.accum*args.batch)),
        num_training_steps=int(args.epoch*len(dataset)/(world_size*args.accum*args.batch))
    )

    # TensorBoard: Logging
    writer=SummaryWriter()
    step_global=0

    # Training
    for epoch in range(args.epoch):
        # Train Phase
        model_ddp.train()
        sampler.set_epoch(epoch)

        loss_train=0
        optimizer.zero_grad()

        for step, (data, label) in enumerate(dataloader):
            data=data.to(rank)
            label=label.to(rank)
            
            # Get Past-Key-Values
            past_key_values=model_ddp(batch_size=data.shape[0], device=rank)
            # Forward: Base (Pre-Trained) LM
            outputs=pretrained(input_ids=data, labels=label, past_key_values=past_key_values)
            
            loss=outputs[0]/args.accum
            loss.backward()
            loss_train+=loss.item()
            
            if (step+1)%args.accum==0:
                step_global+=1
                
                if rank==0:
                    # TensorBoard
                    writer.add_scalar(
                        f'loss_train/Prefix-Tuning_preseqlen{args.preseqlen}_hidden{args.hidden}_batch{args.batch*args.accum*world_size}_lr{args.lr}_epoch{args.epoch}',
                        loss_train,
                        step_global
                    )

                # Set Loss to 0
                loss_train=0

                optimizer.step()
                scheduler.step()
                
                optimizer.zero_grad()
                
    if rank==0:
        # Save Model
        model_ddp.to(torch.device('cpu'))
        torch.save(model_ddp.module, f'./model/Prefix-Tuning_DDP({world_size})_preseqlen{args.preseqlen}_hidden{args.hidden}_batch{args.batch*args.accum*world_size}_lr{args.lr}_epoch{args.epoch}.pt')

def train_attr_algn(device):
    """
    Paper: Attribute Alignment: Controlling Text Generation from Pre-trained Language Models (https://arxiv.org/abs/2103.11070)
    """
    print('\n***** Attribute-Alignment *****')
    print('Method:', args.method)
    if args.method!='A':
        print('Corpus Domain:', args.domain)
    print('Batch Size:', args.batch*args.accum)
    print('Learning Rate:', args.lr)
    print('Epochs:', args.epoch)
    print('Hidden Dimension Size:', args.hidden)
    print('Device:', device)
    print('******************************\n')

    # Load Pre-Trained Tokenizer, LM
    tokenizer, pretrained=load_pretrained(args.base)
    pretrained=pretrained.to(device)

    # Load Dataset
    dataset=DatasetAttrAlgn(path_data=args.dataset,tokenizer=tokenizer)
    # Collate Function: Padding for Same Sequence Length on Same Batch
    if args.method=='A':
        collate_fn=collate_fn_attr_algn(pad_token_id=tokenizer.pad_token_id)
    elif args.method=='AC':
        collate_fn=collate_fn_attr_algn(pad_token_id=tokenizer.pad_token_id, domain=tokenizer.encode(args.domain))
    # DataLoader
    dataloader=DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    # Load Model
    if args.method=='A':
        model=AttrAlgnA(base_config=pretrained.config, hidden_dim=args.hidden)
    elif args.method=='AC':
        model=AttrAlgnAC(base_config=pretrained.config, hidden_dim=args.hidden)
    
    # Optim, Scheduler
    optimizer=AdamW(model.parameters(), lr=args.lr)
    scheduler=get_linear_schedule_with_warmup(
        optimizer=optimizer,
        # 3% of Total Steps
        num_warmup_steps=int(0.03*args.epoch*len(dataset)/(args.accum*args.batch)),
        num_training_steps=int(args.epoch*len(dataset)/(args.accum*args.batch))
    )

    # TensorBoard: Logging
    writer=SummaryWriter()
    step_global=0

    # Training
    for epoch in range(args.epoch):
        # Train Phase
        model.train()
        model.to(device)

        loss_train=0
        optimizer.zero_grad()

        for step, (data, label, attr, domain) in enumerate(dataloader):
            data=data.to(device)
            label=label.to(device)
            attr=attr.to(device)

            # Attribute(s) Representation from Base (Pre-Trained) LM
            encoded_attr=pretrained(input_ids=attr, use_cache=True)['past_key_values']

            if args.method=='A':
                # Transform Attribute(s) Representation via Alignment Function
                past_key_values=model(input_key_values=encoded_attr)
            elif args.method=='AC':
                # Corpus Domain Representation from Base (Pre-Trained) LM
                encoded_domain=pretrained(input_ids=domain.to(device), use_cache=True)['past_key_values']
                # Transform Attribute(s) Representation via Alignment Function
                past_key_values=model(domain_key_values=encoded_domain, attr_key_values=encoded_attr)

            # Forward: Base (Pre-Trained) LM
            outputs=pretrained(input_ids=data, labels=label, past_key_values=past_key_values)
            
            loss=outputs[0]/args.accum
            loss.backward()
            loss_train+=loss.item()
            
            if (step+1)%args.accum==0:
                step_global+=1
                
                # TensorBoard
                writer.add_scalar(
                    f'loss_train/Attribute-Alignment({args.method})_hidden{args.hidden}_batch{args.batch*args.accum}_lr{args.lr}_epoch{args.epoch}',
                    loss_train,
                    step_global
                )

                # Set Loss to 0
                loss_train=0

                optimizer.step()
                scheduler.step()
                
                optimizer.zero_grad()
                
        # Save Model
        model.to(torch.device('cpu'))
        torch.save(model, f'./model/Attribute-Alignment({args.method})_hidden{args.hidden}_batch{args.batch*args.accum}_lr{args.lr}_epoch{epoch+1}of{args.epoch}.pt')

def train_control_prefixes(device):
    """
    Paper: Control Prefixes for Text Generation (https://arxiv.org/abs/2110.08329)
    """
    print('\n***** Control-Prefixes *****')
    print('Batch Size:', args.batch*args.accum)
    print('Learning Rate:', args.lr)
    print('Epochs:', args.epoch)
    print('Prefix Sequence Length:', args.preseqlen)
    print('Hidden Dimension Size:', args.hidden)
    print('Device:', device)
    print('******************************\n')

    # Load Pre-Trained Tokenizer, LM
    tokenizer, pretrained=load_pretrained(args.base)
    pretrained=pretrained.to(device)

    # Load Dataset
    dataset=DatasetControlPrefixes(path_data=args.dataset,tokenizer=tokenizer)
    # Collate Function: Padding for Same Sequence Length on Same Batch
    collate_fn=collate_fn_control_prefixes(pad_token_id=tokenizer.pad_token_id)
    # DataLoader
    dataloader=DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    
    # Config of Control-Codes for Control-Prefixes
    control_config=load_config_for_control_prefixes(args.dataset)
    # Load Model
    model=ControlPrefixes(base_config=pretrained.config, control_config=control_config, preseqlen=args.preseqlen, hidden_dim=args.hidden)
    
    # Optim, Scheduler
    optimizer=AdamW(model.parameters(), lr=args.lr)
    scheduler=get_linear_schedule_with_warmup(
        optimizer=optimizer,
        # 3% of Total Steps
        num_warmup_steps=int(0.03*args.epoch*len(dataset)/(args.accum*args.batch)),
        num_training_steps=int(args.epoch*len(dataset)/(args.accum*args.batch))
    )

    # TensorBoard: Logging
    writer=SummaryWriter()
    step_global=0

    # Training
    for epoch in range(args.epoch):
        # Train Phase
        model.train()
        model.to(device)

        loss_train=0
        optimizer.zero_grad()

        for step, (data, label, control) in enumerate(dataloader):
            data=data.to(device)
            label=label.to(device)
            
            # Get Past-Key-Values
            past_key_values=model(batch_size=data.shape[0], control=control, device=device)
            # Forward: Base (Pre-Trained) LM
            outputs=pretrained(input_ids=data, labels=label, past_key_values=past_key_values)
            
            loss=outputs[0]/args.accum
            loss.backward()
            loss_train+=loss.item()
            
            if (step+1)%args.accum==0:
                step_global+=1
                
                # TensorBoard
                writer.add_scalar(
                    f'loss_train/Control-Prefixes_preseqlen{args.preseqlen}_hidden{args.hidden}_batch{args.batch*args.accum}_lr{args.lr}_epoch{args.epoch}',
                    loss_train,
                    step_global
                )

                # Set Loss to 0
                loss_train=0

                optimizer.step()
                scheduler.step()
                
                optimizer.zero_grad()
                
        # Save Model
        model.to(torch.device('cpu'))
        torch.save(model, f'./model/Control-Prefixes_preseqlen{args.preseqlen}_hidden{args.hidden}_batch{args.batch*args.accum}_lr{args.lr}_epoch{epoch+1}of{args.epoch}.pt')

def train_prefix_tuning(device):
    """
    Paper: Prefix-Tuning: Optimizing Continuous Prompts for Generation (https://arxiv.org/abs/2101.00190)
    """
    print('\n***** Prefix-Tuning *****')
    print('Batch Size:', args.batch*args.accum)
    print('Learning Rate:', args.lr)
    print('Epochs:', args.epoch)
    print('Prefix Sequence Length:', args.preseqlen)
    print('Hidden Dimension Size:', args.hidden)
    print('Device:', device)
    print('******************************\n')

    # Load Pre-Trained Tokenizer, LM
    tokenizer, pretrained=load_pretrained(args.base)
    pretrained=pretrained.to(device)

    # Load Dataset
    dataset=DatasetPrefixTuning(path_data=args.dataset,tokenizer=tokenizer)
    # Collate Function: Padding for Same Sequence Length on Same Batch
    collate_fn=collate_fn_prefix_tuning(pad_token_id=tokenizer.pad_token_id)
    # DataLoader
    dataloader=DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)

    # Load Model (on Device)
    model=PrefixTuning(base_config=pretrained.config, preseqlen=args.preseqlen, hidden_dim=args.hidden)
    
    # Optim, Scheduler
    optimizer=AdamW(model.parameters(), lr=args.lr)
    scheduler=get_linear_schedule_with_warmup(
        optimizer=optimizer,
        # 3% of Total Steps
        num_warmup_steps=int(0.03*args.epoch*len(dataset)/(args.accum*args.batch)),
        num_training_steps=int(args.epoch*len(dataset)/(args.accum*args.batch))
    )

    # TensorBoard: Logging
    writer=SummaryWriter()
    step_global=0

    # Training
    for epoch in range(args.epoch):
        # Train Phase
        model.train()
        model.to(device)

        loss_train=0
        optimizer.zero_grad()

        for step, (data, label) in enumerate(dataloader):
            data=data.to(device)
            label=label.to(device)
            
            # Get Past-Key-Values
            past_key_values=model(batch_size=data.shape[0], device=device)
            # Forward: Base (Pre-Trained) LM
            outputs=pretrained(input_ids=data, labels=label, past_key_values=past_key_values)
            
            loss=outputs[0]/args.accum
            loss.backward()
            loss_train+=loss.item()
            
            if (step+1)%args.accum==0:
                step_global+=1
                
                # TensorBoard
                writer.add_scalar(
                    f'loss_train/Prefix-Tuning_preseqlen{args.preseqlen}_hidden{args.hidden}_batch{args.batch*args.accum}_lr{args.lr}_epoch{args.epoch}',
                    loss_train,
                    step_global
                )

                # Set Loss to 0
                loss_train=0

                optimizer.step()
                scheduler.step()
                
                optimizer.zero_grad()
                
        # Save Model
        model.to(torch.device('cpu'))
        torch.save(model, f'./model/Prefix-Tuning_preseqlen{args.preseqlen}_hidden{args.hidden}_batch{args.batch*args.accum}_lr{args.lr}_epoch{epoch+1}of{args.epoch}.pt')

def main():
    if args.device=='gpu' and torch.cuda.is_available():
        # Multi-GPU
        if args.ddp=='True':
            world_size=torch.cuda.device_count()

            if args.model=='prefix-tuning':
                mp.spawn(train_ddp_prefix_tuning, args=(world_size,), nprocs=world_size, join=True)
            elif args.model=='control-prefixes':
                mp.spawn(train_ddp_control_prefixes, args=(world_size,), nprocs=world_size, join=True)
            elif args.model=='attr-algn':
                mp.spawn(train_ddp_attr_algn, args=(world_size,), nprocs=world_size, join=True)
            else:
                print('Wrong Model Name!')

            return
        # Single GPU
        else:
            device=torch.device('cuda:0')
    else:
        # CPU
        device=torch.device('cpu')

    if args.model=='prefix-tuning':
        train_prefix_tuning(device=device)
    elif args.model=='control-prefixes':
        train_control_prefixes(device=device)
    elif args.model=='attr-algn':
        train_attr_algn(device=device)
    else:
        print('Wrong Model Name!')

if __name__=='__main__':
    main()
