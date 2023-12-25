mkdir "../run/test_cube"
cd "../build"

l0 -v -l 0.01 -i ../data/examples/cube7.obj -o ../run/test_cube/vertex_lambda_1e-2.obj

l0 -e -l 0.006 -i ../data/examples/cube7.obj -o ../run/test_cube/edge_lambda_6e-3.obj

l0 -a -l 0.006 -i ../data/examples/cube7.obj -o ../run/test_cube/area_lambda_6e-3.obj

l0 -a -l 0.006 -i ../data/examples/cube7.obj -o ../run/test_cube/area_lambda_6e-3_regualtion.obj -r

l0 -a -i ../data/examples/cube7.obj -o ../run/test_cube/area_lambda_auto.obj

pause