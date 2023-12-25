@echo off
mkdir "../build"
cd "../build"
cmake .. -G "MinGW Makefiles"
::directly use `cmake ..` will generate VS project on windows
make
pause