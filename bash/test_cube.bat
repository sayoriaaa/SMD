mkdir "../run/test_cube"
mkdir "../run/test_cube/log0"
cd "../build"

l0 ../data/examples/cube7.obj ../run/test_cube/vertex_lambda_1e-2.obj -v -l 0.01 
::l0 ../data/examples/cube7.obj ../run/test_cube/edge_lambda_6e-3.obj -e -l 0.006
l0 ../data/examples/cube7.obj ../run/test_cube/area_lambda_6e-3.obj -a -l 0.006  --log ../run/test_cube/log0
l0 ../data/examples/cube7.obj ../run/test_cube/area_lambda_6e-3_regualtion.obj -a -l 0.006 -r -rg 1
::l0 ../data/examples/cube7.obj ../run/test_cube/area_lambda_auto.obj  -a 
bilateral ../data/examples/cube7.obj ../run/test_cube/bilateral.obj --k_ring 1 -i 20
bilateral-norm ../data/examples/cube7.obj ../run/test_cube/bilateral_norm.obj -i 20 -s 0.7 --update_iter 100

metrics ../data/examples/cube7.obj --gt_file ../data/examples/cube.obj --ahd --aad --oep
metrics ../run/test_cube/vertex_lambda_1e-2.obj --gt_file ../data/examples/cube.obj --ahd --aad --oep 
metrics ../run/test_cube/area_lambda_6e-3.obj --gt_file ../data/examples/cube.obj --ahd --aad --oep
metrics ../run/test_cube/area_lambda_6e-3_regualtion.obj --gt_file ../data/examples/cube.obj --ahd --aad --oep 
metrics ../run/test_cube/bilateral.obj --gt_file ../data/examples/cube.obj --ahd --aad --oep 
metrics ../run/test_cube/bilateral_norm.obj --gt_file ../data/examples/cube.obj --ahd --aad --oep  

pause