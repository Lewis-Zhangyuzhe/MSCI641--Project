# msci641

#### Author
SIHAN WU
20781178
s357wu@uwaterloo.ca

YUZHE ZHANG
20774276
y2877zha@uwaterloo.ca

## Introduction
This directory contains all the code we have for this FNC stance detection project for now. The best result is produced by [BERT](https://colab.research.google.com/drive/1Ag9dsRPP5T7X1InQmOOFeUJzhBLQX2R9) using colab. The other models and scripts are as following:

- Result plotting and BERT feature extraction description: ./Project.ipynb
        
The extracted features(CLS token representation) are first store in .jsonl file in ./fnc/data. The .jsonl is further processed to produced .npy, which is used by GradientBoost. Detailed processing please check the code.
        
- BERT CLS Token + GradientBoost : ./fnc-1-baseline/grad_bert.py
- Original baseline GradientBoost: ./fnc-1-baseline/fnc_kfold.py