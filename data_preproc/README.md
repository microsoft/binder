# Credits

The data preprocessing code is adapted from [DyGIE](https://github.com/luanyi/DyGIE/tree/master/preprocessing).

# Requirements

* python3
* perl
* nltk (for stanford pos tagger)
* java (for stanford tools)
* task datasets (see below)

# Links to tasks/data sets

* ACE 2004 (https://catalog.ldc.upenn.edu/LDC2005T09)
* ACE 2005 (https://catalog.ldc.upenn.edu/LDC2006T06)
* CoNLL 2003 (We use the preprocessed version from [Yu et al., 2020](https://github.com/juntaoy/biaffine-ner/issues/16))

# Usage


## Download Stanford Core NLP & POS tagger

```bash
cd common
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
wget http://nlp.stanford.edu/software/stanford-postagger-2015-04-20.zip
unzip stanford-corenlp-full-2015-04-20.zip
unzip stanford-postagger-2015-04-20.zip
cd ..
```

## Copy and convert each corpus

Please set the environment variables for the directories, or directly put the directories in the following commands beforehand.

### ACE 2004

```bash
cp -r ${RAW_ACE2004_DIR}/*/ENGLISH ace2004/
cd ace2004
bash run.sh
mkdir -p ../data/ace04/train
mkdir -p ../data/ace04/test
python ace2json.py
bash convert.sh
rm -r corpus fixed result text
cd ..
python convert_to_hf_ds_format.py data/ace05/train.json ${ACE2004}/train.json
python convert_to_hf_ds_format.py data/ace05/dev.json ${ACE2004}/dev.json
python convert_to_hf_ds_format.py data/ace05/test.json ${ACE2004}/test.json
```

### ACE 2005

```bash
cp -r ${RAW_ACE2005_DIR}/*/English ace2005/
cd ace2005
bash run.sh
mkdir -p ../data/ace05/
python ace2json.py
rm -r corpus fixed result text
cd ..
python convert_to_hf_ds_format.py data/ace05/train.json ${ACE2005}/train.json
python convert_to_hf_ds_format.py data/ace05/dev.json ${ACE2005}/dev.json
python convert_to_hf_ds_format.py data/ace05/test.json ${ACE2005}/test.json
```


### CoNLL 2003
```bash
python convert_to_hf_ds_format.py conll2003/train.json ${CoNLL2003}/train.json --task conll2003
python convert_to_hf_ds_format.py conll2003/dev.json ${CoNLL2003}/dev.json --task conll2003
python convert_to_hf_ds_format.py conll2003/test.json ${CoNLL2003}/test.json --task conll2003
```

If you want to use other datasets, please convert them into the same format as above.