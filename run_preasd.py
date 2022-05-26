#!/usr/bin/env python3
# coding: utf-8
"""
Created on Sat May 14 21:35:50 2022

@author: alexandr brut-brulyako brutbrulyako@gmail.com

script creates scoring for look-a-like preasd job sellers.
scores would be added in  IVOLGA.preasd_recomendations
run log in IVOLGA.runlog

requirements:
python>=3.7
verticapy==0.9.0
vertica_python>=1.0.1
pandas==1.4.2
numpy==1.19.2
sklearn>=0.23.2
xgboost==1.3.3
pandarallel==1.6.1
"""

N_JOBS = 10 # число паралельных процесов в питоне
N_REC = 1000 # число рекомендуемых селлеров на запуск

# collect dataset
import collect_dataset_preasd

# make recomendations
import  recomends_preasd
