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
option(VI_SME2 "Enable ARM SME2 matrix kernels when supported by compiler" ON)

include(CheckCXXSourceCompiles)

# Portable kernel sources: Eigen (GEMM), pocketfft (FFT). Both header-only.
# They are not vendored; tools/fetch_third_party.sh downloads the pinned versions.
set(VI_THIRD_PARTY ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
foreach(need Eigen/Core xsimd/xsimd.hpp pocketfft_hdronly.h)
  if(NOT EXISTS ${VI_THIRD_PARTY}/${need})
    message(FATAL_ERROR "third_party/${need} is missing. Run tools/fetch_third_party.sh first.")
  endif()
endforeach()

add_library(voiceisolate
  src/model.cc
  src/stream.cc
  src/wav.cc
  src/fft.cc
  src/weights.cc
  src/pool.cc
  src/vecmath.cc
  src/gemm_int8.cc
  src/gemm_sme.cc
)
target_include_directories(voiceisolate
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${VI_THIRD_PARTY})
target_compile_options(voiceisolate PRIVATE -O3 -fno-math-errno)
if(VI_NATIVE)
  target_compile_options(voiceisolate PRIVATE -march=native)
endif()
if(VI_SME2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  set(CMAKE_REQUIRED_FLAGS "-march=armv8.7-a+sme+sme2+fp16+bf16")
  check_cxx_source_compiles("
    #include <arm_sme.h>
    __arm_locally_streaming __arm_new(\"za\") void test_sme() { svzero_za(); }
    int main() { return 0; }
  " VI_COMPILER_SUPPORTS_SME2)
  set(CMAKE_REQUIRED_FLAGS "")
  if(VI_COMPILER_SUPPORTS_SME2)
    set_source_files_properties(src/gemm_sme.cc PROPERTIES
      COMPILE_OPTIONS "-march=armv8.7-a+sme+sme2+fp16+bf16"
      COMPILE_DEFINITIONS "VI_HAS_SME2=1"
    )
    message(STATUS "ARM SME2 matrix kernels enabled")
  endif()
  set(CMAKE_REQUIRED_FLAGS "-march=armv8.7-a+fp16+fp16fml")
  check_cxx_source_compiles("
    #include <arm_neon.h>
    float16x8_t test_f16(float16x8_t a, float16x8_t b, float16x8_t c) { return vfmaq_f16(a, b, c); }
    int main() { return 0; }
  " VI_COMPILER_SUPPORTS_FP16)
  set(CMAKE_REQUIRED_FLAGS "")
  if(VI_COMPILER_SUPPORTS_FP16)
    set_source_files_properties(src/vecmath.cc PROPERTIES
      COMPILE_OPTIONS "-march=armv8.7-a+fp16+fp16fml"
    )
    message(STATUS "ARM NEON FP16 vector kernels enabled")
  endif()
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
