@echo off
mkdir "../build"
cd "../build"
cmake .. -G "MinGW Makefiles"
::directly use `cmake ..` will generate VS project on windows
mingw32-make
pause