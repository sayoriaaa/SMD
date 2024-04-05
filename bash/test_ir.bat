:: sayoriaaa 2024/3/30
@echo off
:: ONLY NEED TO CHANGE BELOW

set level=1
:: noise sigma = 0.1 * level
set list=40359 40447 40843 41969 43960
set mul=0.2
:: model index in thingi10k, make sure stored in data/examples/

:: ONLY NEED TO CHANGE BELOW

set work_dir=../run/ir%level%/
set gt_dir=../data/examples/

set work_dirg=run/ir%level%/
set gt_dirg=data/examples/
::globally visit
set metric_path=%work_dir%metric.txt
set time_path=%work_dir%time.txt
set noise_sigma=0.%level%
set noise_path=%work_dir%noise.obj


mkdir "%work_dir%"
cd "../build"

::empty both files
echo. > %metric_path%
echo. > %time_path%

(for %%a in (%list%) do (
   noise %gt_dir%%%a.obj %work_dir%%%a.obj -f %noise_sigma% -g > nul
)) 

(for %%a in (%list%) do (
   echo %%a
   echo NAME:%%aedge >> %time_path%
   l0 %work_dir%%%a.obj %work_dir%%%aedge.obj -e --mul %mul% >> %time_path%

   echo NAME:%%aarea >> %time_path%
   l0 %work_dir%%%a.obj %work_dir%%%aarea.obj -a --mul %mul%  >> %time_path%

   echo NAME:%%aarea_r >> %time_path%
   l0 %work_dir%%%a.obj %work_dir%%%aarea_r.obj -a -r --mul %mul%  >> %time_path%

   echo NAME:%%a >> %metric_path%
   echo DENOISED:%work_dirg%%%a >> %metric_path%
   echo GT:%gt_dirg%%%a >> %metric_path%
   metrics %work_dir%%%a.obj --gt_file %gt_dir%%%a.obj --ahd --aad --oep >> %metric_path%

   echo NAME:%%aedge >> %metric_path%
   echo DENOISED:%work_dirg%%%aedge >> %metric_path%
   echo GT:%gt_dirg%%%a >> %metric_path%
   metrics %work_dir%%%aedge.obj --gt_file %gt_dir%%%a.obj --ahd --aad --oep >> %metric_path%

   echo NAME:%%aarea >> %metric_path%
   echo DENOISED:%work_dirg%%%aarea >> %metric_path%
   echo GT:%gt_dirg%%%a >> %metric_path%
   metrics %work_dir%%%aarea.obj --gt_file %gt_dir%%%a.obj --ahd --aad --oep >> %metric_path%

   echo NAME:%%aarea_r >> %metric_path%
   echo DENOISED:%work_dirg%%%aarea_r >> %metric_path%
   echo GT:%gt_dirg%%%a >> %metric_path%
   metrics %work_dir%%%aarea_r.obj --gt_file %gt_dir%%%a.obj --ahd --aad --oep >> %metric_path%
)) 
echo Finished

pause