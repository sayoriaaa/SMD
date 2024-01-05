if exist "%CUDA_PATH%" (
    REM see https://stackoverflow.com/questions/43113508/math-functions-hpp-not-found-when-using-cuda-with-eigen
    mklink /H "%CUDA_PATH%\include\math_functions.hpp" "%CUDA_PATH%\include\crt\math_functions.hpp" 
    REM remove above soft link 
    REM rmdir /q /s "%CUDA_PATH%\include\math_functions.hpp"     
) else (
    echo CUDA is not installed
    pause
    exit
)