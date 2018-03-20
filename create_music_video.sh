#! /usr/bin/zsh

# TODO port to python
# Arguments: folder to images, sound file
ffmpeg -r 24 -f image2 -s 1920x1080 -pattern_type glob -i $1/'*.png' -vcodec libx264 -crf 24  $1/video.mp4
ffmpeg -i $1/video.mp4 -i $2 -c:v copy -c:a aac -strict experimental -shortest $1/music_video.mp4
