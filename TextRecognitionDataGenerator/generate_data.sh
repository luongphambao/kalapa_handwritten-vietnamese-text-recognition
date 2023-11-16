#find all file in trdg/fonts/custom/ bash script
count=45
for file in trdg/fonts/custom/*.ttf; do

    python3 trdg/run.py  -c 1000 -w 12 -f 128  -l vi -dt dict.txt -ft $file -na 2 --output_dir out$count
    count=$((count+1))
done