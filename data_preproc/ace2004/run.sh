#!/bin/bash

# ace apf, sgm to ann, txt
mkdir result
for i in ENGLISH/*/*.SGM
do
    echo $i
    tmp_file=`echo $i | sed -e 's/.SGM/_APF.XML/g'`
    if [ -f "${tmp_file}" ];
    then
        python3 ace2ann.py `echo $i | sed -e 's/.SGM/_APF.XML/g'` $i result/`basename $i .SGM`.txt > result/`basename $i .SGM`.ann
    else
        python3 ace2ann.py `echo $i | sed -e 's/.SGM/_A.XML/g'` $i result/`basename $i .SGM`.txt > result/`basename $i .SGM`.ann
    fi
done

# split & parse
mkdir text
java -cp ".:../common/stanford-corenlp-full-2015-04-20/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -props ../common/props_ssplit -outputFormat conll -filelist train_list -outputDirectory text/
# conll to text
for i in text/*.conll
do
    python3 ../common/conll2txt.py $i > text/`basename $i .txt.conll`.split.txt
done

# adjust offsets
for i in text/*.split.txt
do
    echo $i && python3 ../common/standoff.py result/`basename $i .split.txt`.txt result/`basename $i .split.txt`.ann $i > text/`basename $i .split.txt`.split.ann
done

# fix sentence split errors
mkdir fixed
for i in text/*.split.txt
do
    python3 ../common/fix_sentence_break.py $i text/`basename $i .split.txt`.split.ann fixed/`basename $i .split.txt`.txt
done

# parse ssplit-fixed text
java -cp ".:../common/stanford-corenlp-full-2015-04-20/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -props ../common/props_fixed -outputFormat conll -filelist train_list_fixed -outputDirectory fixed/
for i in fixed/*.conll
do
    python3 ../common/conll2txt.py $i > fixed/`basename $i .txt.conll`.split.txt
done

# collect data
mkdir corpus
cd corpus
cp ../result/*.txt .
cp ../result/*.ann .
cp ../fixed/*.split.txt .
cd ..

# adjust offsets
for i in corpus/*.split.txt
do
    echo $i && python3 ../common/standoff.py corpus/`basename $i .split.txt`.txt corpus/`basename $i .split.txt`.ann $i > corpus/`basename $i .split.txt`.split.ann
done

# conll to so
for i in fixed/*.split.txt
do
    echo $i && perl ../common/dep2so.prl fixed/`basename $i .split.txt`.txt.conll $i > corpus/`basename $i .txt`.stanford.so
done

# split data
for i in 0 1 2 3 4
do
    mkdir corpus/train${i}
    mkdir corpus/test${i}
    for j in `cat split/cv${i}`
    do
        mv corpus/$j* corpus/test${i}/
    done
done
for i in 0 1 2 3 4
do
    for j in 0 1 2 3 4
    do
        if [ $i != $j ]; then
           cp corpus/test${i}/* corpus/train${j}/
        fi
    done
done
