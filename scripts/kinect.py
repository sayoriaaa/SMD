import argparse
import os
import subprocess
import platform
import shutil

dataset_dict = {
    "v1": "Kinect_v1",
    "v2": "Kinect_v2",
    "f": "Kinect_Fusion",
    "s": "Synthetic",
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', default=None, help='job name (default: kinect_00x)')
    parser.add_argument('--denoise_command', 
                        default='l0 -a ', 
                        help='denoise command')
    parser.add_argument('--dataset', 
                        default='s', 
                        help='which dataset (Kinect-v1/v2/fusion Syhthetic)')
    return parser.parse_args()

def traverse_obj_files(dataset):
    DATASET = dataset_dict[dataset]
    gt_path = []
    noise_path = []
    gt_name = []
    noise_name = []

    path = os.path.join(os.getcwd(), "data", DATASET, 'test', 'noisy')
    file_list = os.listdir(path) 
    for cnt, file_name in enumerate(file_list):
        noise_name.append(file_name)
        file_path = os.path.join(path, file_name) 
        noise_path.append(file_path)  

    path = os.path.join(os.getcwd(), "data", DATASET, 'test', 'original')
    file_list = os.listdir(path) 
    for cnt, file_name in enumerate(file_list):
        gt_name.append(file_name)
        file_path = os.path.join(path, file_name) 
        gt_path.append(file_path)          
    return gt_path, gt_name, noise_path, noise_name

def create_folder(job_name=None):
    path = os.path.join(os.getcwd(), "run")
    if job_name!=None:
        root = os.path.join(path, job_name)
        os.makedirs(root)
        print(f'Created job: {job_name}')
        return root

    base_folder_name = 'kinect'
    folder_number = 0

    while True:
        folder_name = f'{base_folder_name}_{folder_number:03d}'  # 格式化文件夹名，例如 kinect_001
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
    gt_path, gt_name, noise_path, noise_name = traverse_obj_files(args.dataset)
    # create root folder
    root = create_folder(job_name=args.job_name)
    # create [ gt | noise | denoised ]
    gt_folder = os.path.join(root, 'gt')
    noise_folder = os.path.join(root, 'noise')
    denoised_folder = os.path.join(root, 'denoised')

    
    os.makedirs(gt_folder)
    os.makedirs(noise_folder)
    for cnt, i in enumerate(gt_path): 
        gt_file = os.path.join(gt_folder, gt_name[cnt])
        noise_file = os.path.join(noise_folder, noise_name[cnt])
        denoised_file = os.path.join(denoised_folder, gt_name[cnt])

        gt_files.append(gt_file)
        noise_files.append(noise_file)
        denoised_files.append(denoised_file)

        shutil.copy2(i, gt_file) # init root/gt
        shutil.copy2(noise_path[cnt], noise_file) # init root/noise
        

    # init root/denoised
    os.makedirs(denoised_folder)
    for i in range(len(gt_files)):
        command = os.getcwd()+'/build/'+args.denoise_command+' -i {} -o {}'.format(
                                                        noise_files[i], 
                                                        denoised_files[i])
        subprocess.run(command, shell=True)