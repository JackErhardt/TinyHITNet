# echo "Hello"

# listList="square100 square200"
# modeList="same w h"
# padList="-1 0 32 64 128"

listList="medium small"
modeList="same w h"
padList="-1 0 32 64 128"

for list in $listList; do
	for mode in $modeList; do
		for pad in $padList; do
			touch "sweep/${list}${mode}${pad}.txt"
			case $mode in
				"same")
				roi_w_pad=$pad
				roi_h_pad=$pad
				;;
				"w")
				roi_w_pad=$pad
				roi_h_pad=0
				;;
				"h")
				roi_w_pad=0
				roi_h_pad=$pad
				;;
			esac
			echo "&&& Running ${list} ${mode} ${roi_w_pad} ${roi_h_pad} ..."
			python3 eval.py \
				--model HITNet_KITTI \
				--ckpt ckpt/hitnet_kitti.ckpt \
				--max_disp 192 \
				--data_type_val KITTI2012 KITTI2015 \
				--data_root_val /z/erharj/kitti/2012/training /z/erharj/kitti/2015/training \
				--data_list_val lists/kitti2012_val24_${list}.list lists/kitti2015_val20_${list}.list \
				--data_augmentation 1 \
				--roi_w_pad ${roi_w_pad} \
				--roi_h_pad ${roi_h_pad} > "sweep/${list}${mode}${pad}.txt"
		done
	done
done