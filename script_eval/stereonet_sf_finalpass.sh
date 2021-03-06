python3 eval.py \
	--model StereoNet \
	--ckpt ckpt/stereonet_sf_finalpass.ckpt \
	--max_disp 192 \
	--data_type_val SceneFlow \
	--data_root_val /z/erharj/sceneflow \
	--data_list_val lists/sceneflow_search_val_small.list \
	--data_augmentation 0 \
	--roi_padding 128 \
	# --data_list_val lists/sceneflow_search_val.list \
	# --data_list_val lists/sceneflow_search_val_medium.list \
	# --data_list_val lists/sceneflow_search_val_small.list \