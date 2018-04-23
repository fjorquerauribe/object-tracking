if [ "$4" = "detector" ]; then
    python ../tests/test_gmphd.py -i ./data/$1/$2/$3/img1/ -g ./data/$1/$2/$3/gt/gt.txt
elif [ "$4" = "public" ]; then
    python ../tests/test_gmphd.py -i ./data/$1/$2/$3/img1/ -g ./data/$1/$2/$3/gt/gt.txt -d ./data/$1/$2/$3/det/det.txt
else
    python ../tests/test_gmphd.py -i ./data/$1/$2/$3/img1/ -g ./data/$1/$2/$3/gt/gt.txt -d ./data/detections/FRCNN/$1/$2/$3.txt
fi