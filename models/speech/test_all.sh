for fname in $1/*
do
    echo $fname
    python test_speech.py $fname 2>log
done
