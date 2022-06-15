python3 predict.py \
	--images /z/erharj/kitti/2015/training/image_2/000153_10.png /z/erharj/kitti/2015/training/image_3/000153_10.png \
	--roi 136 188 970 1152 \
	--model HITNet_KITTI \
	--ckpt ckpt/hitnet_kitti.ckpt \
	--roi_padding 0 \