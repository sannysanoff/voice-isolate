cmake_minimum_required(VERSION 3.16)
project(voice_isolate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VI_NATIVE "Tune for the build machine (-march=native). Desktop only" ON)
option(VI_WARNINGS "Enable warnings" ON)

# Portable kernel sources: Eigen (GEMM), pocketfft (FFT). Both header-only.
# They are vendored so the build needs no network.
set(VI_THIRD_PARTY ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_library(voiceisolate
  src/model.cc
  src/stream.cc
  src/wav.cc
  src/fft.cc
  src/weights.cc
  src/pool.cc
  src/vecmath.cc
)
target_include_directories(voiceisolate
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${VI_THIRD_PARTY})
target_compile_options(voiceisolate PRIVATE -O3 -fno-math-errno)
if(VI_NATIVE)
  target_compile_options(voiceisolate PRIVATE -march=native)
endif()
find_package(Threads REQUIRED)
target_link_libraries(voiceisolate PUBLIC Threads::Threads)

if(VI_WARNINGS)
  target_compile_options(voiceisolate PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

add_executable(voice-isolate-cli apps/vi_cli.cc)
target_link_libraries(voice-isolate-cli PRIVATE voiceisolate)
target_compile_options(voice-isolate-cli PRIVATE -O3)
if(VI_NATIVE)
  target_compile_options(voice-isolate-cli PRIVATE -march=native)
endif()

add_executable(vi_tests tests/vi_tests.cc)
target_link_libraries(vi_tests PRIVATE voiceisolate)
target_compile_options(vi_tests PRIVATE -O3)

enable_testing()
add_test(NAME vi_tests COMMAND vi_tests ${CMAKE_CURRENT_SOURCE_DIR})
