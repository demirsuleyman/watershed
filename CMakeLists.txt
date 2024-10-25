cmake_minimum_required(VERSION 3.22.1)
project(watershed)

# Find CUDA
find_package(CUDA REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Set CUDA and OpenCV include directories
include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES
        main.cpp  # Replace with your source files
)

# Compile the main application
add_executable(watershed ${SOURCES})

# Link the CUDA and OpenCV libraries
target_link_libraries(watershed ${CUDA_LIBRARIES} ${OpenCV_LIBS})