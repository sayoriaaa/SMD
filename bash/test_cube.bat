:: sayoriaaa 2024/3/30
@echo off
:: ONLY NEED TO CHANGE BELOW

set level=1
:: noise sigma = 0.1 * level
:: model index in thingi10k, make sure stored in data/examples/

:: ONLY NEED TO CHANGE BELOW
:: sayoriaaa 2024/3/30
set gt_path=../data/examples/cube.obj
set work_dir=../run/cube%level%/
set log_dir=%work_dir%log
set metric_path=%work_dir%metric.txt
set noise_sigma=0.%level%
set mul=1
:: do not change noise_path
set noise_path=%work_dir%noise.obj


mkdir "%work_dir%"
mkdir "%log_dir%"
cd "../build"

echo. > %metric_path%

noise %gt_path% %work_dir%noise.obj -f %noise_sigma% -g -i
L0 %noise_path% %work_dir%vert.obj -v 
L0 %noise_path% %work_dir%edge.obj -e --mul %mul%
L0 %noise_path% %work_dir%area.obj -a   --log %log_dir% --mul %mul%
L0 %noise_path% %work_dir%area_r.obj -a -r --mul %mul%
L0 %noise_path% %work_dir%area_rf.obj  -a -r -rg 1 --mul %mul%
BF %noise_path% %work_dir%BF.obj --k_ring 1 -i 20
BNF %noise_path% %work_dir%BNF.obj -i 20 -s %noise_sigma% --update_iter 100
BGF %noise_path% %work_dir%BGF.obj -i 20 -s %noise_sigma% --update_iter 100  

:: evaluate
::del "%metric_path%" do it manually, if run again
set list=noise vert edge area area_r area_rf BF BNF BGF
(for %%a in (%list%) do (
   echo %%a >> %metric_path%
   echo NAME:%%a >> %metric_path%
   echo DENOISED:%work_dir%%%a >> %metric_path%
   echo GT:%gt_path% >> %metric_path%
   metrics %work_dir%%%a.obj --gt_file %gt_path% --ahd --aad --oep >> %metric_path%

)) 
pause