#!/bin/bash

for f in *.avi; do 
  newname=`echo $f | sed -e 's/\.avi//'`;
  ffmpeg -i $f -c:a aac -b:a 128k -c:v libx264 -crf 23 ${newname}.mp4; 
done
