#!/bin/bash

FILES=`ls datasets/*.in`
OUT="out.txt"
CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of (logical) cores
NREPS=5

rm -rf "./sse/*"

for fin in $FILES ;
do
    echo $fin
    otname=`sed 's/datasets\///' <<< "$fin"`
    RES="sse/res-$otname.txt"
    touch $RES
    echo $fin > $RES
    # echo -e "p\tt1\tt2\tt3\tt4\tt5" >> $RES
    echo -e "t1\tt2\tt3\tt4\tt5" >> $RES
    for p in `seq $CORES`; do
        # echo -n -e "$p\t" >> $RES
        for rep in `seq $NREPS`; do
            OMP_NUM_THREADS=$p ./omp-skyline < $fin > $OUT
            TIME=`tail -n 1 $OUT`
            echo -n -e "$TIME\t" >> $RES
        done
        rm $OUT
        echo "" >> $RES
    done
    echo "" >> $RES
done
