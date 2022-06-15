python3 eval.py \
	--model HITNet_KITTI \
	--ckpt ckpt/hitnet_kitti_crop_scratch_small.ckpt \
	--max_disp 192 \
	--data_type_val KITTI2012 KITTI2015 \
	--data_root_val /z/erharj/kitti/2012/training /z/erharj/kitti/2015/training \
	--data_list_val lists/kitti2012_val24_small.list lists/kitti2015_val20_small.list \
	--data_augmentation 1 \
	--roi_padding 128 \
	# --ckpt ckpt/hitnet_kitti.ckpt \
	# --ckpt ckpt/hitnet_kitti_crop_scratch_medium.ckpt \
	# --ckpt ckpt/hitnet_kitti_crop_scratch_small.ckpt \
	# --data_list_val lists/kitti2012_val24.list lists/kitti2015_val20.list \
	# --data_list_val lists/kitti2012_val24_medium.list lists/kitti2015_val20_medium.list \
	# --data_list_val lists/kitti2012_val24_small.list lists/kitti2015_val20_small.list \
