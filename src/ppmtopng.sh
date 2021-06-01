#!/bin/bash

for dir in `ls`;
do
    # echo "$dir"
    for img in $dir/*.ppm;
    do
        name=`echo "$img" | cut -d'.' -f1`
        ffmpeg -i "$img" "${name}.png" -hide_banner -loglevel error &
    done
done