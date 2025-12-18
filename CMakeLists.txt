cmake_minimum_required(VERSION 3.13)
set(PICO_BOARD pico_w CACHE STRING "Board type")
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(pico_sdk_import.cmake)
project(tinyml-wine-classifier C CXX ASM)
pico_sdk_init()

# Add pico-tflmicro library
add_subdirectory(lib/pico-tflmicro EXCLUDE_FROM_ALL)

include_directories(${CMAKE_SOURCE_DIR}/lib)

add_executable(${PROJECT_NAME}  
        main.c
        tflm_wrapper.cpp
)


target_link_libraries(${PROJECT_NAME} 
        pico_stdlib
        hardware_sync
        pico-tflmicro
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR} 

        )

pico_enable_stdio_usb(${PROJECT_NAME} 1)
pico_enable_stdio_uart(${PROJECT_NAME} 0)

pico_add_extra_outputs(${PROJECT_NAME})

