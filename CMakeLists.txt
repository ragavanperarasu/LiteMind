cmake_minimum_required(VERSION 3.20)

project(LiteMind VERSION 0.1.0 LANGUAGES CXX)

# Keep the language level explicit so all supported generators compile the same API.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(LiteMind
    src/main.cpp
    src/Cli.cpp
    src/Tokenizer.cpp
    src/Config.cpp
    src/Logger.cpp
    src/Tensor.cpp
    src/SafeTensor.cpp
    src/WeightReader.cpp
    src/CpuTensor.cpp
    src/MoeRouter.cpp
    src/Sampler.cpp
    src/KvCache.cpp
    src/Attention.cpp
    src/Model.cpp
    src/DeepSeekRunner.cpp
)

target_include_directories(LiteMind PRIVATE include C:/msys64/mingw64/include/openblas)
target_link_libraries(LiteMind PRIVATE C:/msys64/mingw64/lib/libopenblas.dll.a)

# Apply useful warnings without turning compiler-specific options into part of the codebase.
if(MSVC)
    target_compile_options(LiteMind PRIVATE /W4 /permissive-)
else()
    target_compile_options(LiteMind PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Put binaries in a predictable directory for both single- and multi-config generators.
set_target_properties(LiteMind PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
