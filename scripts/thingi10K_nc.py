import argparse
import os
import openmesh as om
import subprocess
import platform





DATASET = "Thingi10K_name_and_category"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', default='./', help='specific category of Thingi10K (default: all category)')
    parser.add_argument('--job_name', default=None, help='job name (default: thingi10k_00x)')
    parser.add_argument('--noise_factor', default=0.3, help='noise sigma')
    parser.add_argument('--noise_command', 
                        default='-g -f 0.3', 
                        help='denoise command')
    parser.add_argument('--denoise_command', 
                        default='l0 -a ', 
                        help='denoise command')
    return parser.parse_args()

def traverse_stl_files(folder):
    stl_path = []
    stl_name = []
    path = os.path.join(os.getcwd(), "data", DATASET, folder)
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.stl'):
                stl_file_name = os.path.splitext(os.path.basename(file))[0]
                stl_file_path = os.path.join(path, root, file)
                stl_file_path = os.path.normpath(stl_file_path)  # 规范化路径
                stl_path.append(stl_file_path)

                stl_number = ''.join(filter(str.isdigit, stl_file_name))
                stl_txt = f"{stl_number}.txt"  
                stl_txt_path = os.path.join(path, root, stl_txt)
                stl_txt_path = os.path.normpath(stl_txt_path)  # 规范化路径

                if os.path.exists(stl_txt_path):
                    with open(stl_txt_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        stl_name.append(content) 
                else:
                    print(f"Dataset currupted, file {stl_txt_path} does not exist.")
                #print(stl_txt_path) 
    return stl_path, stl_name

def create_folder(job_name=None):
    path = os.path.join(os.getcwd(), "run")
    if job_name!=None:
        root = os.path.join(path, job_name)
        os.makedirs(root)
        print(f'Created job: {job_name}')
        return root

    base_folder_name = 'thingi10k'
    folder_number = 0

    while True:
        folder_name = f'{base_folder_name}_{folder_number:03d}'  # 格式化文件夹名，例如 thingi10k_001
        root = os.path.join(path, folder_name)
        if not os.path.exists(root):
            os.makedirs(root)
            print(f'Created job: {folder_name}')
            return root
        folder_number += 1
    

if __name__ == "__main__":
    win = True if platform.system() == 'Windows' else False
    args = parse_arguments()
    gt_files = []
    noise_files = []
    denoised_files = []
    # find all meshes
    file, name = traverse_stl_files(args.folder)
    for i in range(len(name)):
        print(file[i])
        print(name[i])
    # create root folder
    root = create_folder(job_name=args.job_name)
    # create [ gt | noise | denoised ]
    gt_folder = os.path.join(root, 'gt')
    noise_folder = os.path.join(root, 'noise')
    denoised_folder = os.path.join(root, 'denoised')

    # init root/gt
    os.makedirs(gt_folder)
    for cnt,i in enumerate(file): 
        # openmesh may output std::cerr in this process
        # therefore the corresponding mesh might have topological problems
        # to discuss this issue later 
        gt_file = os.path.join(gt_folder, f'{cnt:06d}.obj')
        noise_file = os.path.join(noise_folder, f'{cnt:06d}.obj')
        denoised_file = os.path.join(denoised_folder, f'{cnt:06d}.obj')

        gt_files.append(gt_file)
        noise_files.append(noise_file)
        denoised_files.append(denoised_file)

        mesh = om.read_trimesh(i)
        om.write_mesh(gt_file, mesh)
        

    # init root/noise
    os.makedirs(noise_folder)   
    for i in range(len(gt_files)):
        command = os.getcwd()+'/build/noise '+args.noise_command+' -i {} -o {}'.format(
                                                        gt_files[i], 
                                                        noise_files[i])
        subprocess.run(command, shell=True)
    # init root/denoised
    os.makedirs(denoised_folder)
    for i in range(len(gt_files)):
        command = os.getcwd()+'/build/'+args.denoise_command+' -i {} -o {}'.format(
                                                        noise_files[i], 
                                                        denoised_files[i])
        subprocess.run(command, shell=True)





    
        

