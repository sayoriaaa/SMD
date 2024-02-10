import os
import openmesh as om

root = os.path.join(os.getcwd(), "data", 'PrintedDataset_r')
os.mkdir(root)
os.mkdir(os.path.join(root, 'noisy'))
os.mkdir(os.path.join(root, 'gt'))

gt_path, gt_name, noise_path, noise_name = [], [], [], []

path = os.path.join(os.getcwd(), "data", 'PrintedDataset', 'noisy')
file_list = os.listdir(path) 
for cnt, file_name in enumerate(file_list):
    file_path = os.path.join(path, file_name) 
    mesh = om.read_trimesh(file_path) 
    om.write_mesh(os.path.join(root, 'noisy', file_name), mesh)

path = os.path.join(os.getcwd(), "data", 'PrintedDataset', 'gt')
file_list = os.listdir(path) 
for cnt, file_name in enumerate(file_list):
    file_path = os.path.join(path, file_name) 
    mesh = om.read_trimesh(file_path) 
    om.write_mesh(os.path.join(root, 'gt', file_name), mesh)