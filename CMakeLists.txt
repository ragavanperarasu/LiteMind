cmake_minimum_required(VERSION 3.20)

project(LiteMind VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Inference is unusable in a debug build, so default to an optimised one when
# the generator has no configuration of its own.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LITEMIND_NATIVE_ARCH "Tune for this machine's instruction set (-march=native)" ON)

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# ── The engine, built once and shared by the executable and the tests ─────────
add_library(litemind_core STATIC
    src/Json.cpp
    src/Config.cpp
    src/Logger.cpp
    src/Tensor.cpp
    src/MappedFile.cpp
    src/SafeTensor.cpp
    src/WeightStore.cpp
    src/ExpertCache.cpp
    src/Threading.cpp
    src/Gemm.cpp
    src/Rope.cpp
    src/KvCache.cpp
    src/Attention.cpp
    src/MoeRouter.cpp
    src/Sampler.cpp
    src/Tokenizer.cpp
    src/DeepSeekRunner.cpp
)

target_include_directories(litemind_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Threads REQUIRED)
target_link_libraries(litemind_core PUBLIC Threads::Threads)

# ── Optimisation ─────────────────────────────────────────────────────────────
# The BF16 kernels are hand-vectorised for AVX2, but the compiler needs to be
# told the target supports it. Without this the scalar fallback runs, which is
# roughly eight times slower.
if(MSVC)
    target_compile_options(litemind_core PUBLIC /W4 /permissive- /bigobj)
    target_compile_options(litemind_core PUBLIC $<$<NOT:$<CONFIG:Debug>>:/O2>)
    if(LITEMIND_NATIVE_ARCH)
        check_cxx_compiler_flag("/arch:AVX2" LITEMIND_MSVC_AVX2)
        if(LITEMIND_MSVC_AVX2)
            target_compile_options(litemind_core PUBLIC /arch:AVX2)
        endif()
    endif()
else()
    target_compile_options(litemind_core PUBLIC -Wall -Wextra -Wpedantic)
    target_compile_options(litemind_core PUBLIC $<$<NOT:$<CONFIG:Debug>>:-O3>)
    if(LITEMIND_NATIVE_ARCH)
        check_cxx_compiler_flag("-march=native" LITEMIND_HAS_NATIVE)
        if(LITEMIND_HAS_NATIVE)
            target_compile_options(litemind_core PUBLIC -march=native)
        else()
            # Apple Silicon and other non-x86 targets reject -march=native.
            check_cxx_compiler_flag("-mcpu=native" LITEMIND_HAS_MCPU_NATIVE)
            if(LITEMIND_HAS_MCPU_NATIVE)
                target_compile_options(litemind_core PUBLIC -mcpu=native)
            endif()
        endif()
    endif()
endif()

# MinGW needs an explicit large-address-aware 64-bit toolchain to map a 9 GB
# shard. Catch a 32-bit compiler at configure time rather than at runtime.
check_cxx_source_compiles("
    #include <cstddef>
    static_assert(sizeof(void*) == 8, \"64-bit required\");
    int main() { return 0; }
" LITEMIND_IS_64_BIT)
if(NOT LITEMIND_IS_64_BIT)
    message(FATAL_ERROR
        "LiteMind needs a 64-bit compiler: a 32-bit process cannot memory-map a "
        "multi-gigabyte checkpoint. On MSYS2 use the MINGW64 shell, not MINGW32.")
endif()

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(LiteMind src/main.cpp src/Cli.cpp)
target_link_libraries(LiteMind PRIVATE litemind_core)

set_target_properties(LiteMind PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

message(STATUS "LiteMind ${PROJECT_VERSION}: ${CMAKE_BUILD_TYPE} build, "
               "native tuning ${LITEMIND_NATIVE_ARCH}")
