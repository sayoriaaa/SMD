import pyvista
import os
import numpy as np
import math

# https://docs.pyvista.org/version/stable/examples/02-plot/scalar-bars
sargs = dict(height=0.2, vertical=True, position_x=0.05, position_y=0.05, n_labels=2)

def visual_norm(f1, f_ref):
    mesh = pyvista.read(f1)
    mesh_ref = pyvista.read(f_ref)

    mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
    mesh_ref.compute_normals(cell_normals=True, point_normals=False, inplace=True)

    diff = []
    for i in range(mesh['Normals'].shape[0]):
        t = mesh['Normals'][i].dot(mesh_ref['Normals'][i])
        t = min(max(t,0),1)
        t = math.acos(t)
        diff.append(t*180/math.pi)
    mesh.cell_data[''] =  diff
    return mesh

def visual_vert(f1, f_ref):
    mesh = pyvista.read(f1)
    mesh_ref = pyvista.read(f_ref)

    diff = []
    for i in range(mesh.points.shape[0]):
        v = mesh.points[i]
        t = 1e10
        for j in range(mesh_ref.points.shape[0]):
            temp = np.linalg.norm(mesh.points[i]-mesh_ref.points[j])
            if t > temp:
                t = temp
        diff.append(t)
    mesh.point_data[''] =  diff
    return mesh


def example_subplot(meshes, fig_name='img.png', titles=None, sargs=sargs):
    l = len(meshes)
    p = pyvista.Plotter(shape=(1, l), off_screen=True, border=False)
    for i, m in enumerate(meshes):
        p.subplot(0, i)
        p.add_mesh(
            m,
            scalar_bar_args=sargs,
            cmap="jet"
        )
        if titles!=None:
            #p.add_title(titles[i])
            p.add_text(titles[i], font='times', font_size=18)

    if titles!=None:
        sz = [512*l, 512+140]
    else:
        sz = [512*l, 512]
    p.show(window_size=sz, screenshot=fig_name)


if __name__=='__main__':
    f0 = os.path.join(os.getcwd(), 'mesh', '10.obj')
    f1 = os.path.join(os.getcwd(), 'mesh', '20.obj')
    f2 = os.path.join(os.getcwd(), 'mesh', '30.obj')
    f3 = os.path.join(os.getcwd(), 'mesh', '40.obj')
    f = os.path.join(os.getcwd(), 'mesh', 'cube.obj')

    #p = pyvista.Plotter()
    m1 = visual_norm(f0, f)
    m2 = visual_norm(f1, f)
    m3 = visual_norm(f2, f)
    m4 = visual_norm(f3, f)
    example_subplot([m1, m2, m3, m4], titles=['iter:10', 'iter:20', 'iter:30', 'iter:40'])