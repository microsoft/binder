#!/bin/bash

cd ..

for part in 0 1 2 3 4
do
    python -u convert_to_hf_ds_format.py data/ace04/train/${part}.json ../data/ace04/train/${part}.json
    python -u convert_to_hf_ds_format.py data/ace04/test/${part}.json ../data/ace04/test/${part}.json
done

cd ace2004