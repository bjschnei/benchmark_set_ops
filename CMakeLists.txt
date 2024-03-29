cmake_minimum_required(VERSION 3.26)

project(
  SetOps
  DESCRIPTION "Benchmark set operations [union, intersection, difference"
  LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Include FetchContent module
include(FetchContent)

# CRoaring
FetchContent_Declare(
  roaring
  GIT_REPOSITORY "https://github.com/RoaringBitmap/CRoaring.git"
  GIT_TAG 3946d399a9a71e2fe890326b21ecbb45283ea3fe
)
FetchContent_MakeAvailable(roaring)
set(CPP_ROARING_HEADERS ${roaring_SOURCE_DIR}/cpp/roaring64map.hh
                        ${roaring_SOURCE_DIR}/cpp/roaring.hh)
file(COPY ${CPP_ROARING_HEADERS}
     DESTINATION ${roaring_SOURCE_DIR}/include/roaring)

# Absl
set(ABSL_PROPAGATE_CXX_STD ON)
FetchContent_Declare(
  absl
  GIT_REPOSITORY "https://github.com/abseil/abseil-cpp.git"
  GIT_TAG 20240116.1)
FetchContent_MakeAvailable(absl)

# BitMagic
FetchContent_Declare(
  bm
  GIT_REPOSITORY "https://github.com/tlk00/BitMagic.git"
  GIT_TAG v7.13.4)
FetchContent_MakeAvailable(bm)
include_directories(${bm_SOURCE_DIR}/src)

# Benchmark
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY "https://github.com/google/benchmark.git"
  GIT_TAG v1.8.3)
FetchContent_MakeAvailable(benchmark)

add_library(sets INTERFACE sets.h)
add_executable(set_ops main.cc)
target_link_libraries(
  set_ops
  PUBLIC absl::any_invocable
         absl::random_random
         absl::status
         absl::statusor
         absl::strings
         benchmark::benchmark
         bm
         gtest
         roaring
         sets)
