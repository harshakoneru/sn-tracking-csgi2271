for file in /scratch/hk3820/ByteTrack/datasets/SoccerNet/test/*/ ; do 
file="${file%/}" # strip trailing slash
file="${file##*/}"
# echo "$file is the directoryname without slashes"
python tools/demo_track_no_gt.py image -f exps/example/mot/yolox_x_ch_hk3820.py \
-c YOLOX_outputs/yolox_x_ch_hk3820/best_ckpt.pth.tar --fp16 --fuse --save_result \
--path /scratch/hk3820/ByteTrack/datasets/SoccerNet/test/$file/img1/
done
