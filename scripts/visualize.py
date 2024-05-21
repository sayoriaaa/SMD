import pyvista
import os
import numpy as np
import math
import copy

class Sample:
    def __init__(self, name) -> None:
        self.name = name
        # pyvista mesh
        self.noise_mesh = None
        self.denoised_mesh = None
        self.gt_mesh = None
        # metrics 
        self.AAD = None
        self.AHD = None
        self.OFP = None
        self.time = None

    def visual_norm(self):
        mesh = copy.deepcopy(self.denoised_mesh)
        mesh_ = self.gt_mesh

        mesh.compute_normals(cell_normals=True, point_normals=False, inplace=True)
        mesh_.compute_normals(cell_normals=True, point_normals=False, inplace=True)

        diff = []
        for i in range(mesh['Normals'].shape[0]):
            t = mesh['Normals'][i].dot(mesh_['Normals'][i])
            t = min(max(t,0),1)
            t = math.acos(t)
            diff.append(t*180/math.pi)
        mesh.cell_data[''] =  diff
        return mesh

    def visual_vert(self):
        mesh = copy.deepcopy(self.denoised_mesh)
        mesh_ = self.gt_mesh

        diff = []
        for i in range(mesh.points.shape[0]):
            v = mesh.points[i]
            t = 1e10
            for j in range(mesh_.points.shape[0]):
                temp = np.linalg.norm(mesh.points[i]-mesh_.points[j])
                if t > temp:
                    t = temp
            diff.append(t)
        mesh.point_data[''] =  diff
        return mesh

    def show(self, type):
        mesh = None
        if type==-2:
            mesh = self.noise_mesh
        if type==-1:
            mesh = self.gt_mesh
        if type==0:
            mesh = self.denoised_mesh
        if type==1:
            mesh = self.visual_norm()
        if type==2:
            mesh = self.visual_vert()
        return mesh

def clean_file(f):
    # this function is called in global scope
    if f.startswith('../'):
        f = f[3:]
    if f.endswith('.obj'):
        return f 
    else:
        return f+'.obj'
    
class SampleUnion:
    def parse(self, line):
        # NAME:'xxx \n' -> 'xxx'
        # return line.split()[0].split(':')[1]
        return line.split(':')[1].split()[0]

    def __init__(self, job_name) -> None:
        # load results
        work_dir = './run/{}/'.format(job_name)
        assert os.path.exists(work_dir), "Please ensure this job exists in ./run"

        metrics_file = work_dir + 'metric.txt'
        time_file = work_dir + 'time.txt'
        assert os.path.exists(metrics_file), "Please ensure metric.txt exists in this project"

        f=open(metrics_file)
        data = f.readlines() 
        f.close()  

        results = {}
        outer = 0
        name = None

        while(outer<len(data)):
            line = data[outer]
            if line.startswith('NAME'):
                name = self.parse(line)
                results[name] = Sample(name)
            elif line.startswith('NOISE'):
                noise_file = self.parse(line)            
                results[name].noise_mesh = pyvista.read(clean_file(noise_file))
            elif line.startswith('DENOISED'):
                denoised_file = self.parse(line)            
                results[name].denoised_mesh = pyvista.read(clean_file(denoised_file))
            elif line.startswith('GT'):
                gt_file = self.parse(line)
                results[name].gt_mesh = pyvista.read(clean_file(gt_file))
            elif line.startswith('AAD'):
                results[name].AAD = float(self.parse(line))
            elif line.startswith('AHD'):
                results[name].AHD = float(self.parse(line))
            elif line.startswith('OEP'):
                results[name].OEP = float(self.parse(line))
            else:
                pass
            outer += 1

        if os.path.exists(time_file):
            f=open(time_file)
            data = f.readlines() 
            f.close()  

            outer = 0
            sample = None

            for line in data:
                if line.startswith('NAME'):
                    name = self.parse(line)
                    sample = results[name]
                if line.startswith('Execution'):
                    sample.time = float(self.parse(line))
        self.results = results

    def show(self, canvas, fig_name='img.png', mesh_vis_type=None, 
             titles=None, sargs=None, show_bar=True, show_metrics=None, 
             camera_position=None, zoom=1):
        # resemble_samples
        n = len(canvas)
        m = len(canvas[0])
        samples_ = []
        for i in range(n):
            row = []
            for j in range(m):
                name = canvas[i][j]
                row.append(self.results[name])
            samples_.append(row)

        # init mesh_vis_type
        if mesh_vis_type==None:
            mesh_vis_type = [[0 for j in range(m)] for i in range(n)]
        if sargs==None:
            # https://docs.pyvista.org/version/stable/examples/02-plot/scalar-bars
            sargs = dict(height=0.2, vertical=True, position_x=0.05, position_y=0.05, n_labels=2)

        # draw
        p = pyvista.Plotter(shape=(n, m), off_screen=True, border=False)

        for i in range(n):
            for j in range(m):
                sss = samples_[i][j]
                p.subplot(i, j)
                mesh = sss.show(mesh_vis_type[i][j])
                p.add_mesh(
                    mesh,
                    scalar_bar_args=sargs,
                    cmap="jet",
                    show_scalar_bar=show_bar
                )
                p.camera.zoom(zoom)
                if camera_position!=None:
                    p.camera_position = camera_position
                if titles!=None:
                    p.add_text(titles[i][j], font='times', font_size=18, position='upper_edge')
                if show_metrics!=None and show_metrics[i][j]==1:
                    ss = '{:.3f}\n{:.3f}\n{:.3f}'.format(sss.AAD,sss.AHD*1000,sss.OEP)
                    p.add_text(ss, position='lower_right', font_size=15)

        if titles!=None:
            sz = [512*m, 512*n+140]
        else:
            sz = [512*m, 512*n]
        
        p.show(window_size=sz, screenshot=fig_name)
        p.close()

def quick_gen_vis_type(canvas, x):
    """
    e.g. 

    canvas = [["40359", "40359edge", "40359area", "40359area_r"],
                ["40447", "40447edge", "40447area", "40447area_r"],
                ["40843", "40843edge", "40843area", "40843area_r"],
                ["43960", "43960edge", "43960area", "43960area_r"]]

    x = 1
    ->
    [1, 1, 1, 1]
    [1, 1, 1, 1]
    [1, 1, 1, 1]
    [1, 1, 1, 1]

    x = [[0],[1],[0],[1]]
    ->
    [0, 1, 0, 1]
    [0, 1, 0, 1]
    [0, 1, 0, 1]
    [0, 1, 0, 1]

    x = [[0],[1],[0],[1]]
    ->
    [0, 0, 0, 0]
    [1, 1, 1, 1]
    [0, 0, 0, 0]
    [1, 1, 1, 1]

    """
    n = len(canvas)
    m = len(canvas[0])
    mesh_vis_type = [[0 for j in range(m)] for i in range(n)]

    if type(x)==int:
        for i in range(n):
            for j in range(m):
                mesh_vis_type[i][j] = x

    if type(x)==list and type(x[0])==int:
        # row vector
        for i in range(n):
            for j in range(m):
                mesh_vis_type[i][j] = x[j]
    if type(x)==list and type(x[0])==list:
        # row vector
        for i in range(n):
            for j in range(m):
                mesh_vis_type[i][j] = x[i][0]
    return mesh_vis_type