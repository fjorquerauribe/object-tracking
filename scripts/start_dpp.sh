#python ../src/test_dpp.py -i ./data/$1/$2/$3/img1/ -g ./data/$1/$2/$3/gt/gt.txt -d ./data/$1/$2/$3/det/det.txt
python ../src/test_dpp.py -i ./data/$1/$2/$3/img1/ -g ./data/$1/$2/$3/gt/gt.txt -d ./data/DETS_SSD/$1/$2/0.5-0.5/$3.txt -eps $4 -mu $5 -gamma $6
