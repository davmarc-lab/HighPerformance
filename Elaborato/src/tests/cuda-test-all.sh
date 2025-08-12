#/bin/bash

files=`ls datasets/*.in`

ot="cuda-out"
if [ ! -e $ot ] ; then
    mkdir $ot
fi

make omp

OUTFILE="$ot/out.txt"
if [ ! -e $OUTFILE ] ; then
    touch $OUTFILE
fi

OUTDIR="nall"
if [[ ! -e $OUTDIR ]] ; then
    mkdir $OUTDIR
fi

rm $OUTDIR/*

RES="$OUTDIR/res-all-ntimes.txt"

for fin in $files ;
do
    i=0
    echo $fin
    echo $fin >> $RES
    echo -e "t1\tt2\tt3\tt4\tt5" >> $RES
    for i in {1..5} ;
    do
        ./cuda-skyline < $fin > $OUTFILE
        echo `head $OUTFILE -n 2`
        echo -n -e "`tail $OUTFILE -n 1`\t" >> $RES
    done
    echo -e "\n" >>$RES
done

