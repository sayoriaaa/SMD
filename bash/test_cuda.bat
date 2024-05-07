@echo off
:: ONLY NEED TO CHANGE BELOW
set name=block
:: ONLY NEED TO CHANGE ABOVE
set work_dir=../run/test_cuda/
set gt_path=../data/Synthetic/test/original/%name%.obj
set noise_path=../data/Synthetic/test/noisy/%name%_n2.obj

set metric_path=%work_dir%metric.txt
set time_path=%work_dir%time.txt

mkdir "%work_dir%"
cd "../build"

::nvprof is not supported on newest gpu
::https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
::ncu -o profile l0cuda %noise_path% %work_dir%gpu0.obj 
l0cuda %noise_path% %work_dir%gpu0.obj 
pause
