:: sayoriaaa 2024/4/3
@echo off
setlocal enabledelayedexpansion 
set work_dir=../run/selected/

::
::--------------Synthetic--------------
::
::25K     part_Lp 268KB
set Name1=part_Lp
set G1=../data/Synthetic/test/original/%Name1%.obj
set N1=../data/Synthetic/test/noisy/%Name1%_n2.obj

::31K     trim-star 322KB
set Name2=trim-star
set G2=../data/Synthetic/test/original/%Name2%.obj
set N2=../data/Synthetic/test/noisy/%Name2%_n2.obj

::38K     fandisk 419KB
set Name3=fandisk
set G3=../data/Synthetic/train/original/%Name3%.obj
set N3=../data/Synthetic/train/noisy/%Name3%_n2.obj

::47K     octa-flower 731KB
set Name4=octa-flower
set G4=../data/Synthetic/train/original/%Name4%.obj
set N4=../data/Synthetic/train/noisy/%Name4%_n2.obj

::53K     block 543KB
set Name5=block
set G5=../data/Synthetic/test/original/%Name5%.obj
set N5=../data/Synthetic/test/noisy/%Name5%_n2.obj

::84K     fertility 895KB
set Name6=fertility
set G6=../data/Synthetic/test/original/%Name6%.obj
set N6=../data/Synthetic/test/noisy/%Name6%_n2.obj

::125K    joint 1414KB
set Name7=joint
set G7=../data/Synthetic/test/original/%Name7%.obj
set N7=../data/Synthetic/test/noisy/%Name7%_n2.obj

::179K    bunny 2058KB
set Name8=bunny
set G8=../data/Synthetic/train/original/%Name8%.obj
set N8=../data/Synthetic/train/noisy/%Name8%_n2.obj

::300K    chinese_lion 3589KB
set Name9=chinese_lion
set G9=../data/Synthetic/test/original/%Name9%.obj
set N9=../data/Synthetic/test/noisy/%Name9%_n2.obj

::909K     dragon 11437KB
set Name10=dragon
set G10=../data/Synthetic/train/original/%Name10%.obj
set N10=../data/Synthetic/train/noisy/%Name10%_n2.obj

::
::--------------Kinect v1--------------
::

set Name11=big_girl_01
set G11=../data/Kinect_v1/train/original/%Name11%.obj
set N11=../data/Kinect_v1/train/noisy/%Name11%_noisy.obj

set Name12=cone_01
set G12=../data/Kinect_v1/train/original/%Name12%.obj
set N12=../data/Kinect_v1/train/noisy/%Name12%_noisy.obj

set Name13=david_01
set G13=../data/Kinect_v1/train/original/%Name13%.obj
set N13=../data/Kinect_v1/train/noisy/%Name13%_noisy.obj

set Name14=pyramid_01
set G14=../data/Kinect_v1/train/original/%Name14%.obj
set N14=../data/Kinect_v1/train/noisy/%Name14%_noisy.obj

set Name15=boy_01
set G15=../data/Kinect_v1/test/original/%Name15%.obj
set N15=../data/Kinect_v1/test/noisy/%Name15%_noisy.obj

::
::--------------Kinect v2--------------
::

set Name16=big_girl_03
set G16=../data/Kinect_v2/train/original/%Name16%.obj
set N16=../data/Kinect_v2/train/noisy/%Name16%_noisy.obj

set Name17=cone_03
set G17=../data/Kinect_v2/train/original/%Name17%.obj
set N17=../data/Kinect_v2/train/noisy/%Name17%_noisy.obj

set Name18=david_03
set G18=../data/Kinect_v2/train/original/%Name18%.obj
set N18=../data/Kinect_v2/train/noisy/%Name18%_noisy.obj

set Name19=pyramid_03
set G19=../data/Kinect_v2/train/original/%Name19%.obj
set N19=../data/Kinect_v2/train/noisy/%Name19%_noisy.obj

set Name20=boy_03
set G20=../data/Kinect_v2/test/original/%Name20%.obj
set N20=../data/Kinect_v2/test/noisy/%Name20%_noisy.obj

::
::--------------Kinect Fusion--------------
::

set Name21=big_girl
set G21=../data/Kinect_Fusion/train/original/%Name21%.obj
set N21=../data/Kinect_Fusion/train/noisy/%Name21%_noisy.obj

set Name22=cone
set G22=../data/Kinect_Fusion/train/original/%Name22%.obj
set N22=../data/Kinect_Fusion/train/noisy/%Name22%_noisy.obj

set Name23=girl
set G23=../data/Kinect_Fusion/train/original/%Name23%.obj
set N23=../data/Kinect_Fusion/train/noisy/%Name23%_noisy.obj

set Name24=boy02
set G24=../data/Kinect_Fusion/test/original/%Name24%.obj
set N24=../data/Kinect_Fusion/test/noisy/%Name24%_noisy.obj

set Name25=david
set G25=../data/Kinect_Fusion/test/original/%Name25%.obj
set N25=../data/Kinect_Fusion/test/noisy/%Name25%_noisy.obj


::finish setting files
set metric_path=%work_dir%metric.txt
set time_path=%work_dir%time.txt
set exec=l0
set nums=25

mkdir "%work_dir%"
cd "../build"

::empty both files
echo. > %time_path%
echo. > %metric_path%


FOR /L %%i IN (1 1 %nums%) DO  (
   ::show progress
   cls
   echo Progress:%%i/%nums%

   echo NAME:!Name%%i! >> %time_path%
   %exec% !N%%i! %work_dir%!Name%%i!.obj -a >> %time_path% 

   echo NAME:!Name%%i! >> %metric_path%
   echo NOISE:!N%%i! >> %metric_path%
   echo DENOISED:%work_dir%!Name%%i! >> %metric_path%
   echo GT:!G%%i! >> %metric_path%
   metrics %work_dir%!Name%%i!.obj --gt_file !G%%i! --ahd --aad --oep >> %metric_path%
)
echo Finished

pause