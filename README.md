# Controlled Text Generation using Prefix-Tuning on GPT
Prefix-Tuning is a method that prepends continuous *Trainable* prompts (called *Prefix*) to input sequence while freezing Pre-Trained LM. Prefix-Tuning makes time/memory-efficient training and performance improvements over several downstream tasks.

<img src="https://user-images.githubusercontent.com/89329469/151740274-3b756d88-1fbc-4d71-8863-35befd8a143c.jpg" width="50%" height="50%">

I have referenced the following papers:
* Prefix-Tuning: Optimizing Continuous Prompts for Generation ([Li and Liang](https://arxiv.org/abs/2101.00190), 2021)
* Control Prefixes for Text Generation ([Clive et al.](https://arxiv.org/abs/2110.08329), 2021)
* Attribute Alignment: Controlling Text Generation from Pre-trained Language Models ([Yu et al.](https://arxiv.org/abs/2103.11070), 2021)

Reviews on my own [Blog](https://chainsmokers.oopy.io/) (Korean).
## Usage (OS: Ubuntu)
### Dependencies
* pandas
* openpyxl
* pytorch
* transformers (**v4.15.0**)
### Initialization
```bash
git clone THIS_REPO
cd THIS_REPO
# move your dataset file to dataset/
mv YOUR_DATASET dataset/
# MUST create model/ directory where trained models will be saved
mkdir model
```
### Dataset Format (.xlsx)
Check up *datatset/sample_data.xlsx* (AG News). Check space characters in *code* carefully.
|code|record|
|----|----|
|\| Business&nbsp;|Beyond Meat corp. (BYND)..|
|\| Sports&nbsp;|Son (Tottenham FC) scored..|
|\| World \| Business&nbsp;|Due to G7 summit, Nasdaq..|
### Training
Support 3 models: Prefix-Tuning, Control-Prefixes, Attribute-Alignments, but **NOT exactly same as in paper**. Customized for my own needs.
```bash
python train.py --dataset=dataset/sample_data.xlsx \ # Dataset Path
--model=control-prefixes \ # prefix-tuning | control-prefixes | attr-algn
--base=gpt2-large \ # Base (Pre-Trained) LM
--device=gpu \ # Default: GPU. If not available, load model on CPU
--ddp=False # Use PyTorch DDP or not. Default: False
```
### Generation
Use Nucleus (Top-p) Sampling Strategy.
```bash
python generate.py --model=control-prefixes \
--path=model/Control-Prefixes_preseqlen5_hidden512_batch128_lr5e-05_epoch3of5.pt \ # Trained Model Path
--code='| Business ' # Control Code
--prompt='Google' # Prompt Text
```
### Results
```
# Business
Google and Microsoft have agreed to co-develop a browser for smartphones that runs on their respective Web services.
Google Inc. on Tuesday said its (Nasdaq: GOOG) search ads business is facing its second quarterly loss in three months as it struggles to boost ad growth.
Google on Thursday announced a stock split, giving investors the opportunity to invest in its online advertising business.

# Sports
Google News reported that Kobe Bryant will play this season with the Lakers. It was the first confirmation the Mamba would be part of the new squad that opened the season at Oracle Arena.
Google has won the rights to manage Irish rugby league for a second time after the league and its owners reached a deal to sell their rights to United Football League (UFL) for a record \$30 million.
Google has no immediate plans to bid on the ad-buying rights to the Dutch football league, a spokesman said Monday, as a consortium of football companies proposed an unprecedented sale of the lucrative rights.
```

