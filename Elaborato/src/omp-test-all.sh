#/bin/bash

files=`ls ./datasets/*.in`

ot="omp-out"
if [ ! -e $ot ] ; then
    mkdir $ot
fi

make omp

rs="omp-result.txt"
if [ -e $rs ] ; then
    rm $rs
fi
touch $rs

i=0
for fin in $files ;
do
    ((i=$i + 1))
    echo $fin
    ./omp-skyline < $fin > $ot/$ot$i.txt
    echo $fin >> $rs
    head "$ot/$ot$i.txt" -n 2 >> $rs
    tail "$ot/$ot$i.txt" -n 1 >> $rs
    echo "" >> $rs
    echo ""
done
