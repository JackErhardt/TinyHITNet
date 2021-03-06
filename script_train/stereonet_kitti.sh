python3 train.py \
--log_dir logs \
--exp_name stereonet_kitti \
--model StereoNet \
--check_val_every_n_epoch 9 \
--sync_batchnorm True \
--max_steps 2000000 \
--accelerator gpu \
--strategy ddp \
--max_disp 192 \
--optmizer RMS \
--lr 1e-3 \
--lr_decay 14000 0.9 \
--lr_decay_type Step \
--batch_size 1 \
--batch_size_val 1 \
--num_workers 16 \
--num_workers_val 2 \
--data_augmentation 1 \
--data_type_train KITTI2012 KITTI2015 \
--data_root_train /z/erharj/kitti/2012/training /z/erharj/kitti/2015/training \
--data_list_train lists/kitti2012_train170.list lists/kitti2015_train180.list \
--data_size_train 1152 320 \
--data_type_val KITTI2015 \
--data_root_val /z/erharj/kitti/2015/training \
--data_list_val lists/kitti2015_val20.list \
--data_size_val 1242 375 \
--pretrain ckpt/stereonet_sf_finalpass.ckpt \