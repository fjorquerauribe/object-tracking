declare -a datasets=(2DMOT2015 MOT16)
#declare -a datasets=(2DMOT2015)
OUTPUT_DIRECTORY=$1

for dataset in "${datasets[@]}"
do
    mkdir -p results/$OUTPUT_DIRECTORY/$dataset/train/
    while read sequence;
    do
        echo $dataset,$sequence
        /bin/bash $PWD/start_phd.sh $dataset train $sequence 10 > ./results/$OUTPUT_DIRECTORY/$dataset/train/$sequence.txt
    done <./data/$dataset/train/sequences.lst
done
