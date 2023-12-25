import argparse
import os
import openmesh as om
import subprocess
import platform
import shutil





DATASET = "Thingi10K"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', default=5, help='number of stl mesh (n<=9956)')
    parser.add_argument('--job_name', default=None, help='job name (default: thingi10k_00x)')
    parser.add_argument('--noise_factor', default=0.3, help='noise sigma')
    parser.add_argument('--noise_command', 
                        default='-g -f 0.3', 
                        help='denoise command')
    parser.add_argument('--denoise_command', 
                        default='l0 -a ', 
                        help='denoise command')
    return parser.parse_args()

def traverse_stl_files(num):
    stl_path = []
    path = os.path.join(os.getcwd(), "data", DATASET, 'raw_meshes')
    file_list = os.listdir(path) 
    for cnt, file_name in enumerate(file_list):
        print(cnt)
        if cnt == num:
            return stl_path
        file_path = os.path.join(path, file_name)
        if file_name.endswith('.stl'):  
            stl_path.append(file_path)            
    return stl_path

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
    file = traverse_stl_files(int(args.num))
    for i in file:
        print(i)
    # create root folder
    root = create_folder(job_name=args.job_name)
    # create [ gt | noise | denoised ]
    gt_folder = os.path.join(root, 'gt')
    noise_folder = os.path.join(root, 'noise')
    denoised_folder = os.path.join(root, 'denoised')

    # init root/gt
    os.makedirs(gt_folder)
    for cnt,i in enumerate(file): 
        gt_file = os.path.join(gt_folder, f'{cnt:06d}.stl')
        noise_file = os.path.join(noise_folder, f'{cnt:06d}.obj')
        denoised_file = os.path.join(denoised_folder, f'{cnt:06d}.obj')

        gt_files.append(gt_file)
        noise_files.append(noise_file)
        denoised_files.append(denoised_file)

        shutil.copy2(i, gt_file)
        

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