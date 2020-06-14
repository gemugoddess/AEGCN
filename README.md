# AEGCN
This is our Tensorflow implementation for AEGCN:



## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:

tensorflow-gpu 1.13.1 or 1.12.0
numpy 1.17.0
scipy 1.1.0
pandas 0.25.1
python 3.7.0 or 3.6.2

## Examples of Tianchi 
missing rate = n%, e.g, n=99, 0%<=n%<100.0%

parameters in parser.py

* Command
```
python AEGCN.py --dataset tianchi --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 300 --unlabel_rate=0.99 --mtl_dropout=0.2 --lamda2 0.01 --pre_dim2 64
```

## Dataset

We provide two processed datasets: Tiachi (dataset in paper) and lastfm (just for fast debug).

Format of Tianchi: user_id,item_id,olditemcategory,itemcategory

Every line is an interaction. itemcategory is the item attribute used in paper (top 300 brands)

