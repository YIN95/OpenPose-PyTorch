#!/bin/bash

echo "Convert Videos to Images ..."
#  '2017-01-18' '2017-01-25' '2017-02-01'
for folder in '2017-01-14'
do
    echo $folder
    videofolder=/media/ywj/File/D-Data/dancers/$folder/video
    savefolder=/media/ywj/File/D-Data/dancers/$folder/images

    filelist=`ls $videofolder|grep -i '.*avi'`

    for file in $filelist
    do 
        echo $file
        video_path=$videofolder/${file}
        echo $video_path
        python preprocess_vid2imgs.py --videoPath $video_path --savePath $savefolder
    done
done