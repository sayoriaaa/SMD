:: sayoriaaa 2024/04/8
@echo off
setlocal enabledelayedexpansion 
:: ONLY NEED TO CHANGE BELOW

set name=block
set nums=40
set method=0
:: 0 is jacobi; 1 is gauss-seidel; 2 is CG, 3 is BiCGSTAB, 4 is MINRES, 5 is GMRES


:: ONLY NEED TO CHANGE ABOVE
set work_dir=../run/iter-%name%-%method%/
set gt_path=../data/Synthetic/test/original/%name%.obj
set noise_path=../data/Synthetic/test/noisy/%name%_n2.obj
set metric_path=%work_dir%metric.txt
set time_path=%work_dir%time.txt
set exec=l0-iter

mkdir "%work_dir%"
cd "../build"

::empty both files
echo. > %metric_path%
echo. > %time_path%

FOR /L %%i IN (0 1 %nums%) DO  (

   cls
   echo Progress:%%i/%nums%

   echo NAME:%name%%%i >> %time_path%
   %exec% %noise_path% %work_dir%%name%%%i.obj -a --solver_type %method% --set_iter %%i  >> %time_path% 
   echo. >> %time_path%

   echo NAME:%name%%%i >> %metric_path%
   echo DENOISED:%work_dir%%name%%%i.obj >> %metric_path%
   echo GT:%gt_path% >> %metric_path%
   metrics %work_dir%%name%%%i.obj --gt_file %gt_path% --ahd --aad --oep >> %metric_path%
   echo. >> %metric_path%

)
echo Finished

pause