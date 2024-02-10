import argparse
import os
import subprocess
import platform
import shutil
import json
import time

# CNR (learning based method) dataset
CNR_dataset = ["Kinect_v1", "Kinect_v2", "Kinect_Fusion", "Synthetic"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name',
                        default = None)
    parser.add_argument('--denoise_algorithm',
                        default = "l0")
    parser.add_argument('--denoise_args')
    parser.add_argument('--metrics_args',
                        default='--ahd --aad --oep ')
    parser.add_argument('--dataset',
                        default = "Synthetic")
    parser.add_argument('--use_train', 
                        action = "store_true",
                        help = "only for CNR dataset")    
    return parser.parse_args()

def traverse_files(dataset, use_train = False):
    # CNR (learning based method) dataset, including Kinect_v1 Kinect_v2 Kinect_Fusion Synthetic
    gt_path, gt_name, noise_path, noise_name = [], [], [], []

    if dataset in CNR_dataset:
        path = os.path.join(os.getcwd(), "data", dataset, 'test', 'noisy')
        file_list = os.listdir(path) 
        for cnt, file_name in enumerate(file_list):
            noise_name.append(file_name)
            file_path = os.path.join(path, file_name) 
            noise_path.append(file_path)  

        path = os.path.join(os.getcwd(), "data", dataset, 'test', 'original')
        file_list = os.listdir(path) 
        for cnt, file_name in enumerate(file_list):
            gt_name.append(file_name)
            file_path = os.path.join(path, file_name) 
            gt_path.append(file_path)  

        if use_train:
            path = os.path.join(os.getcwd(), "data", dataset, 'train', 'noisy')
            file_list = os.listdir(path) 
            for cnt, file_name in enumerate(file_list):
                noise_name.append(file_name)
                file_path = os.path.join(path, file_name) 
                noise_path.append(file_path)  

            path = os.path.join(os.getcwd(), "data", dataset, 'train', 'original')
            file_list = os.listdir(path) 
            for cnt, file_name in enumerate(file_list):
                gt_name.append(file_name)
                file_path = os.path.join(path, file_name) 
                gt_path.append(file_path) 

    elif dataset == 'PrintedDataset':
        path = os.path.join(os.getcwd(), "data", 'PrintedDataset_r', 'noisy')
        file_list = os.listdir(path) 
        for cnt, file_name in enumerate(file_list):
            noise_name.append(file_name)
            file_path = os.path.join(path, file_name) 
            noise_path.append(file_path)  

        path = os.path.join(os.getcwd(), "data", 'PrintedDataset_r', 'gt')
        file_list = os.listdir(path) 
        for cnt, file_name in enumerate(file_list):
            gt_name.append(file_name)
            file_path = os.path.join(path, file_name) 
            gt_path.append(file_path) 

    elif dataset == "Thingi10K_obj": 
        pass 

    return gt_path, gt_name, noise_path, noise_name

def create_folder(dataset, job_name=None):
    path = os.path.join(os.getcwd(), "run")
    if job_name!=None:
        root = os.path.join(path, job_name)
        os.makedirs(root)
        print(f'Created job: {job_name}')
    else:
        base_folder_name = dataset
        folder_number = 0
        while True:
            folder_name = f'{base_folder_name}_{folder_number:03d}'  # 格式化文件夹名，例如 printdata_001
            root = os.path.join(path, folder_name)
            if not os.path.exists(root):
                os.makedirs(root)
                print(f'Created job: {folder_name}')
                break
            folder_number += 1
    # create [ gt | noise | denoised ]
    gt_folder = os.path.join(root, 'gt')
    noise_folder = os.path.join(root, 'noise')
    denoised_folder = os.path.join(root, 'denoised')

    os.makedirs(gt_folder)
    os.makedirs(noise_folder)
    os.makedirs(denoised_folder)
    return root

def exec_job(gt_path, gt_name, noise_path, noise_name, root, algo, algo_args, metrics_args):
    gt_files = []
    noise_files = []
    denoised_files = []

    gt_folder = os.path.join(root, 'gt')
    noise_folder = os.path.join(root, 'noise')
    denoised_folder = os.path.join(root, 'denoised')

    time_dic = {}

    # copy files
    for cnt, i in enumerate(noise_path): 
        gt_file = os.path.join(gt_folder, noise_name[cnt])
        noise_file = os.path.join(noise_folder, noise_name[cnt])
        denoised_file = os.path.join(denoised_folder, noise_name[cnt]) # one gt file might have several noise file

        gt_files.append(gt_file)
        noise_files.append(noise_file)
        denoised_files.append(denoised_file)

        for ccnt, j in enumerate(gt_name):# one gt file might have several noise file
            if noise_name[cnt].startswith(j[:-4]):
                shutil.copy2(gt_path[ccnt], gt_file) # init root/gt
                break 
        shutil.copy2(noise_path[cnt], noise_file) # init root/noise
    
    for i in range(len(gt_files)):
        # call denoise algorithm
        command = os.getcwd() + '/build/' + algo + ' {} {} '.format(noise_files[i], denoised_files[i])
        if algo_args!=None:
            command = command + algo_args
        
        s_time = time.time()
        print("denoising "+noise_files[i])
        ret = subprocess.run(command, shell=True)
        if ret.returncode!= 0:
            print('job failed!')
            return
        e_time = time.time()
        time_dic[noise_files[i]] = (e_time - s_time)

        # calc denoised result
        command = os.getcwd() + '/build/metrics' + ' {} --gt_file {} '.format(denoised_files[i], gt_files[i]) + metrics_args + ' --logfile {}.txt'.format(denoised_files[i][:-4])
        ret = subprocess.run(command, shell=True)
        if ret.returncode!= 0:
            print('job failed!')
            return
        
    with open(os.path.join(root, 'time.json'),"w") as f:
        json.dump(time_dic,f)
   

if __name__ == "__main__":
    win = True if platform.system() == 'Windows' else False
    args = parse_arguments()
    # find all meshes in data dictionary
    gt_path, gt_name, noise_path, noise_name = traverse_files(args.dataset, use_train=args.use_train)
    # create folders in run dictonary
    root = create_folder(args.dataset, job_name=args.job_name)
    # exec job
    exec_job(gt_path, gt_name, noise_path, noise_name, root, args.denoise_algorithm, args.denoise_args, args.metrics_args)