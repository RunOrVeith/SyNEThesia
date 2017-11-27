#! /usr/bin/zsh

VIDEO_URL=$1
FPS=$2

MY_PATH="`dirname \"$0\"`"
TITLE=`youtube-dl --get-filename -o $MY_PATH'/data/%(title)s/%(title)s.%(ext)s' -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio' $VIDEO_URL`
echo $TITLE
youtube-dl -o "$TITLE" -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio' --prefer-ffmpeg $VIDEO_URL
FOLDER="`dirname $TITLE`"
mkdir "$FOLDER"/frames
mkdir "$FOLDER"/audio
ffmpeg -i $TITLE -vf fps=$FPS $FOLDER/frames/%d.png
ffmpeg -i $TITLE $FOLDER/audio/soundtrack.mp3
