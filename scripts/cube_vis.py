from visualize import *

def resemble_meshes(titles_in, meshes):
    n = len(titles_in)
    m = len(titles_in[0])
    meshes_in = []
    for i in range(n):
        row = []
        for j in range(m):
            name = titles_in[i][j]
            row.append(meshes[name])
        meshes_in.append(row)
    return meshes_in

def parse_metric(file):
    f=open(file)
    data = f.readlines() 
    f.close()  
    print(data) 
    ret = {}
    cnt = 0
    for line in data:
        if cnt==0:
            cur = line[:-2]
            ret[cur] = []
            cnt+=1
        elif cnt==1:
            cnt+=1
            continue
        else:
            ret[cur].append(float(line[4:-1]))
            if cnt==4:
                cnt=0
            else:
                cnt+=1
    return ret

proj_name = 'cube3i'
gt_path = '../data/examples/cube.obj'
work_dir = '../run/{}/'.format(proj_name)
sargs = dict(height=0.2, vertical=True, position_x=0.05, position_y=0.05, n_labels=2)

# cube is noise_mesh
names = ['vert', 'edge', 'area', 'area_r', 'area_rf', 'BF', 'BNF', 'BGF', 'noise']
meshes = {}

for name in names:
    file = work_dir + name + '.obj'
    meshes[name] = visual_norm(file, gt_path)

titles_in = [['noise', 'BF', 'BNF', 'BGF'],
              ['vert', 'edge', 'area','area_r']] # must be nxm

meshes_in = resemble_meshes(titles_in, meshes)
metrics = parse_metric(work_dir+'metric.txt')
my_plot(meshes_in, fig_name='../imgs/{}.png'.format(proj_name), titles=titles_in, sargs=sargs, metrics=metrics)