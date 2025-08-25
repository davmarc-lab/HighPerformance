#!/bin/bash

LIST="1024 2048 4096 6144 8192 10230 12288"

OUTFILE="out.txt"
RES="weak/res-times.txt"

if [[ ! -e weak ]] ; then
    mkdir weak
fi
rm weak/*

for elem in $LIST ; do
    file=`ls ./datasets/ | grep $elem`
    NUM_P=$(($elem / 1000))
    echo $file >> $RES
    for i in {1..5} ; do
        OMP_NUM_THREADS=$NUM_P ./omp-skyline < datasets/$file > $OUTFILE
        echo `head $OUTFILE -n 2`
        echo -n -e "`tail $OUTFILE -n 1`\t" >> $RES
    done
    echo "" >> $RES
    echo "" >> $RES
done

