cmake_minimum_required(VERSION 3.16)
project(JumpDiffusionStrategy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for production
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address -fsanitize=undefined")

# Find required packages
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})

# Main strategy executable
add_executable(jumpdiffusion_strategy model.cpp)
target_link_libraries(jumpdiffusion_strategy Eigen3::Eigen)

# Backtest executable
add_executable(jumpdiffusion_backtest backtest/backtest.cpp)
target_link_libraries(jumpdiffusion_backtest Eigen3::Eigen)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(jumpdiffusion_strategy PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(jumpdiffusion_backtest PRIVATE -Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(jumpdiffusion_strategy PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(jumpdiffusion_backtest PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install targets
install(TARGETS jumpdiffusion_strategy jumpdiffusion_backtest
        RUNTIME DESTINATION bin)
