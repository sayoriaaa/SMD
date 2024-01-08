@echo off
mkdir "../build"
cd "../build"
cmake .. -G "MinGW Makefiles"
::directly use `cmake ..` will generate VS project on windows
mingw32-make

echo Current directory is: %CD%

::cuda program on windows is compiled via nvcc and MSCV
if exist "%CUDA_PATH%" (
    echo CUDA found on %CUDA_PATH%
    echo -- Compiling cuda files     
) else (
    echo CUDA is not installed
    pause
    exit
)

nvcc -o L0cuda ../src/L0/main.cu  ../src/L0/l0.cc -I../src/dependencies/clipp/include -I../src/dependencies/libigl/include -IC:/opt/eigen  -lcusparse -lcusolver -w

pause