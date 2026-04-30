if [$# -ne 1]; then
    ffmpeg -r 10 -pattern_type glob -i "*.png" -pix_fmt yuv420p zmovie.mp4
else
    ffmpeg -r $1 -pattern_type glob -i "*.png" -pix_fmt yuv420p zmovie.mp4
fi
