:: sayoriaaa 2024/3/31
@echo off
setlocal enabledelayedexpansion 
:: ONLY NEED TO CHANGE BELOW

set name=nicolo
set nums=40

::use python script to generate it since
::bat can't suport float operation well
:: ensure |m|=nums
set m1=0.1
set m2=0.2
set m3=0.3
set m4=0.4
set m5=0.5
set m6=0.6
set m7=0.7
set m8=0.8
set m9=0.9
set m10=1.0
set m11=1.1
set m12=1.2
set m13=1.3
set m14=1.4
set m15=1.5
set m16=1.6
set m17=1.7
set m18=1.8
set m19=1.9
set m20=2.0
set m21=2.1
set m22=2.2
set m23=2.3
set m24=2.4
set m25=2.5
set m26=2.6
set m27=2.7
set m28=2.8
set m29=2.9
set m30=3.0
set m31=3.1
set m32=3.2
set m33=3.3
set m34=3.4
set m35=3.5
set m36=3.6
set m37=3.7
set m38=3.8
set m39=3.9
set m40=4.0

:: ONLY NEED TO CHANGE ABOVE
set work_dir=../run/robust_%name%/
set gt_path=../data/Synthetic/test/original/%name%.obj
set noise_path=../data/Synthetic/test/noisy/%name%_n2.obj
set metric_path=%work_dir%metric.txt
set time_path=%work_dir%time.txt
set exec=l0

mkdir "%work_dir%"
cd "../build"

::empty both files
echo. > %metric_path%
echo. > %time_path%

FOR /L %%i IN (1 1 %nums%) DO  (

   cls
   echo Progress:%%i/%nums%

   echo NAME:%name%%%i >> %time_path%
   %exec% %noise_path% %work_dir%%name%%%i.obj -a --mul !m%%i!>> %time_path% 
   echo. >> %time_path%

   echo NAME:%name%%%i >> %metric_path%
   echo DENOISED:%work_dir%%name%%%i.obj >> %metric_path%
   echo GT:%gt_path% >> %metric_path%
   metrics %work_dir%%name%%%i.obj --gt_file %gt_path% --ahd --aad --oep >> %metric_path%
   echo. >> %metric_path%

)
echo Finished

pause