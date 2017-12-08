
# Arguments: folder to images, sound file
ffmpeg -r 25 -f image2 -s 1920x1080 -i $1/%d.png -vcodec libx264 -crf 25  $1/music_video.mp4
ffmpeg -i $1/music_video.mp4 -i $2 -c:v copy -c:a aac -strict experimental -shortest $1/music_video.mp4
