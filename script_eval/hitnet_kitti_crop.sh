python3 eval.py \
	--model HITNet_KITTI \
	--ckpt ckpt/hitnet_kitti_crop.ckpt \
	--max_disp 192 \
	--data_type_val KITTI2012 KITTI2015 \
	--data_root_val /z/erharj/kitti/2012/training /z/erharj/kitti/2015/training \
	--data_list_val lists/kitti2012_val24.list lists/kitti2015_val20.list \
	--data_augmentation 1