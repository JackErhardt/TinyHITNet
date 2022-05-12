if [ $HOSTNAME == "d1z" ]
then
	sed -i 's/\/z\/dataset\//\/data\//' dataset/kitti2012.py
	sed -i 's/\/z\/dataset\//\/data\//' dataset/kitti2015.py
	# sed -i 's/\/z\/dataset\//\/data\//' dataset/sceneflow.py

	sed -i 's/\/z\/dataset\//\/data\//' preprocess/plane_fitting_kitti2012.py
	sed -i 's/\/z\/dataset\//\/data\//' preprocess/plane_fitting_kitti2015.py
	# sed -i 's/\/z\/dataset\//\/data\//' preprocess/plane_fitting_sf.py

	sed -i 's/\/z\/dataset\//\/data\//' script_eval/stereonet_kitti.sh

	sed -i 's/\/z\/dataset\//\/data\//' script_predict/stereonet_kitti.sh

	sed -i 's/\/z\/dataset\//\/data\//' script_train/hitnet_kitti.sh
	# sed -i 's/\/z\/dataset\//\/data\//' script_train/hitnet_sf_finalpass.sh
	sed -i 's/\/z\/dataset\//\/data\//' script_train/stereonet_kitti.sh
	# sed -i 's/\/z\/dataset\//\/data\//' script_train/stereonet_sf_finalpass.sh
fi

if [ $HOSTNAME == "r1z" ]
then

fi