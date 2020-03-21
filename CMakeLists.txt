cmake_minimum_required(VERSION 3.0	FATAL_ERROR)

project(crnn)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT Torch_FOUND)
	message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

# Print some message showing some of them
message(STATUS "Pytorch status:")
message(STATUS "	libraries: ${TORCH_LIBARRIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(CrnnDeploy CrnnDeploy.cpp)
target_link_libraries(CrnnDeploy ${OpenCV_LIBS} "${TORCH_LIBRARIES}")
set_property(TARGET CrnnDeploy PROPERTY CXX_STANDARD 14)
