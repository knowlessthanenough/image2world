this is the guideline to transfer intel realsense d435i depth map to 3d world coordinate(include align depth with rgb image), defult [0,0,0] in world coordinate is the top left inner conner of the chessboard.(tested on window11, jetson nano 4g, AGX orin)
(asume you already have the intrinsic_matrix of the rgb camera, if no please connect the realsense camera than run get_RGB_intrinsic.py located in backup folder) 
1. run live_align.py to get the align image. press "c" to get aligned image for rgb camera, which will store in name test_color_{i}.png and will also have the depth map store in name depth_map_{i}.npy , after that press "q" to quit. (remember to capture more image cause sometime opencv may not find the inner conner)

2. run get_RGB_extrinsics.py to get the extrinsics. if opencv cannot find the conner in the image it will print "Checkerboard not found in the image". otherwise it will show the image with inner conner in it, remember to check if all the inner conner in the correct place. and it will save the rotation and translation in the name of rgb_intrinsic_vector.npy and rgb_translation_vector.npy.

3. run depth2world.py to get the world coordinate. load the depth_map, intrinsic and extrinsic parameter get from above it will return a world coordinate numpy array in shape [720,1280,3] the three channel is the XYZ of in world space.

4. run check_depth.py to check if the world coordinate is correct. load the test_color.png , world_coordinate.npy from the last step and the intrinsic and extrinsic parameter of the rgb camera. it will show the rgb image with XYZ axis draw in the orginal of world coordinate in it. you can click on any point to see the output is correct to the measurement in the real world. 
