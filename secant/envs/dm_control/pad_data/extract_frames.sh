#!/usr/bin/env bash

SIZE=84x84

mkdir -p frames/

for i in $(seq 0 9); do
    ffmpeg -i video$i.mp4 -s $SIZE -vf fps=1 frames/video$i'_'%02d.png
done