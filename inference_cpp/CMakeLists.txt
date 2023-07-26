set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# 專案命名
project(humanSeg)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# 路徑檢查
if(Torch_FOUND)
  message(STATUS "Torch library found!")
  message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}" \n)
  message(STATUS "    PROJECT path: ${PROJECT_SOURCE_DIR}" \n)

else()
  message(FATAL_ERROR "Could not locate Torch" \n)
endif()

if(OpenCV_FOUND)
  message(STATUS "Opencv library found!")
  message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" \n)
  message(STATUS "    PROJECT path: ${PROJECT_SOURCE_DIR}" \n)

else()
  message(FATAL_ERROR "Could not locate OpenCV" \n)
endif()

# include_directories(${TORCH_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# include_directories(
# ${PROJECT_SOURCE_DIR}/include
# )
file(GLOB SRC_DIR
  app/*.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${SRC_DIR})
target_link_libraries(${CMAKE_PROJECT_NAME} "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
