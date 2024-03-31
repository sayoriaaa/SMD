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
    outer = 0
    ext = ['', 'edge', 'area', 'area_r']

    print(len(data))
    while(outer<len(data)):
        line = data[outer]
        name_ = line.split("\"")[1]
        for e in ext:
            name = name_ + e
            print(name)
            ret[name] = []
            outer +=2
            line = data[outer]
            print(line)
            ret[name].append(float(line[4:-1])) # AAD

            outer +=1
            line = data[outer]
            print(line)
            ret[name].append(float(line[4:-1])) # AHD

            outer +=1
            line = data[outer]
            print(line)
            ret[name].append(float(line[4:-1])) # OEP
        print(outer)
        outer += 1
        
    return ret

proj_name = 'ir1'
gt_dir = '../data/examples/'
work_dir = '../run/{}/'.format(proj_name)
sargs = dict(height=0.2, vertical=True, position_x=0.05, position_y=0.05, n_labels=2)

# cube is noise_mesh
names = ["40359", "40447", "40843", "41969", "43960"]
ext = ["edge", "area", "area_r", ""]
meshes = {}

for name in names:
    gt_file = gt_dir + name + '.obj'
    for e in ext:
        file = work_dir + name + e + '.obj'
        meshes[name + e] = visual_norm(file, gt_file)

titles_in = [["40359", "40447", "40843", "41969", "43960"],
             ["40359edge", "40447edge", "40843edge", "41969edge", "43960edge"],
             ["40359area", "40447area", "40843area", "41969area", "43960area"],
             ["40359area_r", "40447area_r", "40843area_r", "41969area_r", "43960area_r"]] # must be nxm

meshes_in = resemble_meshes(titles_in, meshes)
metrics = parse_metric(work_dir+'metric.txt')
my_plot(meshes_in, fig_name='../imgs/{}.png'.format(proj_name), titles=titles_in, sargs=sargs, metrics=metrics)