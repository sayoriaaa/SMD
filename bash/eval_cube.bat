mkdir "../run/test_cube"
cd "../build"
metrics -g ../data/examples/cube.obj -i ../data/examples/cube.obj

metrics -g ../data/examples/cube.obj -i ../data/examples/cube7.obj

metrics -g ../data/examples/cube.obj -i ../run/test_cube/vertex_lambda_1e-2.obj

metrics -g ../data/examples/cube.obj -i ../run/test_cube/edge_lambda_6e-3.obj

metrics -g ../data/examples/cube.obj -i ../run/test_cube/area_lambda_6e-3.obj

metrics -g ../data/examples/cube.obj -i ../run/test_cube/area_lambda_6e-3_regualtion.obj

metrics -g ../data/examples/cube.obj -i ../run/test_cube/area_lambda_auto.obj

pause