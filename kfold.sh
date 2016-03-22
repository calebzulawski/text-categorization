#!/bin/sh
predictions=$(mktemp)
./kfold.py $1 $predictions
perl corpora/analyze.pl $1 $predictions
rm $predictions
