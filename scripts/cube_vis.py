from visualize import *

f0 = '../run/test_cube/vertex_lambda_1e-2.obj'
f1 = '../run/test_cube/area_lambda_6e-3.obj'
f2 = '../run/test_cube/area_lambda_6e-3_regualtion.obj'
f3 = '../run/test_cube/bilateral.obj'
f4 = '../run/test_cube/bilateral_norm.obj'
f5 = '../data/examples/cube7.obj'
f = '../data/examples/cube.obj'

sargs = dict(height=0.2, vertical=True, position_x=0.05, position_y=0.05, n_labels=2)

m0 = visual_norm(f5, f)
m1 = visual_norm(f0, f)
m2 = visual_norm(f1, f)
m3 = visual_norm(f2, f)
m4 = visual_norm(f3, f)
m5 = visual_norm(f4, f)

example_subplot([m0, m1, m2, m3, m4, m5], fig_name='../imgs/gallery.png', titles=['noise', 'L0-vert', 'L0-area', 'L0-area-reg', 'bilateral', 'bilateral-normal'], sargs=sargs)

f0 = os.path.join('../run/test_cube/log0/10.obj')
f1 = os.path.join('../run/test_cube/log0/20.obj')
f2 = os.path.join('../run/test_cube/log0/30.obj')
f3 = os.path.join('../run/test_cube/log0/40.obj')

#p = pyvista.Plotter()
m1 = visual_norm(f0, f)
m2 = visual_norm(f1, f)
m3 = visual_norm(f2, f)
m4 = visual_norm(f3, f)
example_subplot([m1, m2, m3, m4], fig_name='../imgs/L0-area-iter.png', titles=['iter:10', 'iter:20', 'iter:30', 'iter:40'], sargs=sargs)