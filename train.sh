#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

python dataset_extractor.py 
python dataset_generator.py

