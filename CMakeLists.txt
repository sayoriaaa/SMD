cmake_minimum_required(VERSION 3.0.0)
project(SMD)

include_directories("src/dependencies/eigen")
include_directories("src/dependencies/libigl/include")
include_directories("src/dependencies/spectra/include")
include_directories("src/dependencies/clipp/include")

add_executable(l0 "src/L0/main.cc" "src/L0/l0.cc" "src/L0/l0.hpp")
add_executable(noise "src/utils/add_noise.cc")
add_executable(metrics "src/utils/metrics.cc" "src/L0/l0.cc")
add_executable(l1_smooth "src/L1CS/smooth.cc" "src/L1CS/smooth.hpp" )
#add_executable(noise "src/add_noise.cc")
#add_executable(L0min "src/minimal.cc")
#add_executable(c2 "src/smooth.cc")
#add_executable(l0_local "src/l0_local.cc")

