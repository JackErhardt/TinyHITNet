python3 predict.py \
	--images /data/kitti/2015/training/image_2/000001_10.png /data/kitti/2015/training/image_3/000001_10.png \
	--model StereoNet \
	--ckpt logs/stereonet_kitti/version_2/checkpoints/epoch=278-step=48824.ckpt \
	--output /afs/eecs.umich.edu/vlsisp/users/erharj/TinyHITNet/stereonet_kitti.png