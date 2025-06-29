cmake_minimum_required(VERSION 3.16)
project(OrnsteinUhlenbeckStrategy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags for production
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address -fsanitize=undefined")

# Set Eigen include directory
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../external/eigen-3.4.0")

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})

# Main strategy executable
add_executable(ornsteinuhlenbeck_strategy model.cpp)

# Backtest executable
add_executable(ornsteinuhlenbeck_backtest backtest/backtest.cpp)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(ornsteinuhlenbeck_strategy PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(ornsteinuhlenbeck_backtest PRIVATE -Wall -Wextra -Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(ornsteinuhlenbeck_strategy PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(ornsteinuhlenbeck_backtest PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Install targets
install(TARGETS ornsteinuhlenbeck_strategy ornsteinuhlenbeck_backtest
        RUNTIME DESTINATION bin)
