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


def example_subplot(meshes, fig_name='img.png', titles=None, sargs=sargs, show_bar=True):
    l = len(meshes)
    p = pyvista.Plotter(shape=(1, l), off_screen=True, border=False)
    for i, m in enumerate(meshes):
        p.subplot(0, i)
        p.add_mesh(
            m,
            scalar_bar_args=sargs,
            cmap="jet",
            show_scalar_bar=show_bar
        )
        if titles!=None:
            #p.add_title(titles[i])
            p.add_text(titles[i], font='times', font_size=18)

    if titles!=None:
        sz = [512*l, 512+140]
    else:
        sz = [512*l, 512]
    p.show(window_size=sz, screenshot=fig_name)
    p.close()

def my_plot(meshes, fig_name='img.png', titles=None, sargs=sargs, show_bar=True, metrics=None):
    # meshes and titles MUST be nxn [[m,m],[m,m]]
    n = len(meshes)
    m = len(meshes[0])
    p = pyvista.Plotter(shape=(n, m), off_screen=True, border=False)

    for i in range(n):
        for j in range(m):
            p.subplot(i, j)
            p.add_mesh(
                meshes[i][j],
                scalar_bar_args=sargs,
                cmap="jet",
                show_scalar_bar=show_bar
            )
            if titles!=None:
                name = titles[i][j]
                p.add_text(name, font='times', font_size=18)
                if metrics!=None:
                    t = metrics[name]
                    ss = '{:.3f}\n{:.3f}\n{:.3f}'.format(t[0],t[1]*1000,t[2])
                    p.add_text(ss, position='lower_right', font_size=15)

    if titles!=None:
        sz = [512*m, 512*n+140]
    else:
        sz = [512*m, 512*n]
    p.show(window_size=sz, screenshot=fig_name)
    p.close()


def standard(meshes, fig_name='img.png', off=True):
    # first row direct render (noise)
    # second row direct render (denoised)
    # third row difference
    # ----------------
    # meshes is 3 times N
    # ----------------
    l = len(meshes[0])
    p = pyvista.Plotter(shape=(3, l), off_screen=off, border=False)

    for kk in range(2):
    # first & second row
        for i, m in enumerate(meshes[kk]):
            p.subplot(kk, i)
            p.add_mesh(
                m
            )
    # third row
    for i, m in enumerate(meshes[2]):
        p.subplot(2, i)
        p.add_mesh(
            m,
            #scalar_bar_args=sargs,
            cmap="jet"
        )
        
    sz = [512*l, 512*3]
    p.show(window_size=sz, screenshot=fig_name)
    p.close()




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