cmake_minimum_required(VERSION 3.20)

project(LiteMind VERSION 0.2.0 LANGUAGES CXX)

# ── Toolchain requirements ───────────────────────────────────────────────────
# Checked immediately after the compiler is identified, and before any
# find_package call. An unusable compiler otherwise surfaces as a failure deep
# inside an unrelated CMake module, which points at the wrong problem.

if(CMAKE_SIZEOF_VOID_P LESS 8)
    message(FATAL_ERROR
        "LiteMind needs a 64-bit compiler, but ${CMAKE_CXX_COMPILER} targets "
        "${CMAKE_SIZEOF_VOID_P}-byte pointers.\n"
        "A 32-bit process cannot memory-map a multi-gigabyte checkpoint.\n"
        "On MSYS2, build from the MINGW64 shell and use "
        "C:/msys64/mingw64/bin/g++.exe.")
endif()

# The minimum versions that implement the C++20 library features used here:
# std::span, std::bit_cast and designated initialisers.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10)
    message(FATAL_ERROR
        "LiteMind needs GCC 10 or newer, but ${CMAKE_CXX_COMPILER} is GCC "
        "${CMAKE_CXX_COMPILER_VERSION}.\n"
        "\n"
        "This is almost always the old MinGW.org toolchain in C:/MinGW being "
        "found ahead of a modern one on PATH. It is 32-bit and predates C++20.\n"
        "\n"
        "Install MSYS2 and its 64-bit toolchain, then build from the "
        "'MSYS2 MINGW64' shell:\n"
        "    pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja\n"
        "\n"
        "Or point CMake straight at it from any shell:\n"
        "    cmake -S . -B build -DCMAKE_CXX_COMPILER=C:/msys64/mingw64/bin/g++.exe\n"
        "\n"
        "scripts/build.ps1 finds a suitable compiler on its own.")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12)
    message(FATAL_ERROR
        "LiteMind needs Clang 12 or newer, but ${CMAKE_CXX_COMPILER} is "
        "${CMAKE_CXX_COMPILER_VERSION}.")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.29)
    message(FATAL_ERROR
        "LiteMind needs Visual Studio 2019 16.11 or newer for C++20, but this "
        "compiler reports ${CMAKE_CXX_COMPILER_VERSION}.")
endif()

message(STATUS "Toolchain: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} "
               "(${CMAKE_SIZEOF_VOID_P}-byte pointers)")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Inference is unusable in a debug build, so default to an optimised one when
# the generator has no configuration of its own.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LITEMIND_NATIVE_ARCH "Tune for this machine's instruction set (-march=native)" ON)

# A MinGW executable links against libstdc++-6.dll, libgcc_s_seh-1.dll and
# libwinpthread-1.dll by default. Those live in the MSYS2 tree, so running the
# .exe from an ordinary PowerShell prompt fails to start the process at all --
# Windows reports STATUS_DLL_NOT_FOUND and PowerShell usually prints nothing,
# which looks exactly like a program that ran and produced no output. Linking
# the runtime statically makes the executable self-contained and removes a
# failure mode that is very hard to diagnose from the symptom.
option(LITEMIND_STATIC_RUNTIME "Link the C++ runtime statically (recommended on Windows)" ON)

include(CheckCXXCompilerFlag)

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

# ── Runtime linkage ──────────────────────────────────────────────────────────
if(LITEMIND_STATIC_RUNTIME)
    if(MINGW)
        # -static covers libstdc++, libgcc and libwinpthread in one flag.
        target_link_options(litemind_core INTERFACE -static)
    elseif(MSVC)
        # Requires CMP0091, which is the default from CMake 3.15.
        set_property(TARGET litemind_core PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

# ── Executable ───────────────────────────────────────────────────────────────
add_executable(LiteMind src/main.cpp src/Cli.cpp)
target_link_libraries(LiteMind PRIVATE litemind_core)

if(LITEMIND_STATIC_RUNTIME AND MSVC)
    set_property(TARGET LiteMind PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set_target_properties(LiteMind PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

message(STATUS "LiteMind ${PROJECT_VERSION}: ${CMAKE_BUILD_TYPE} build, "
               "native tuning ${LITEMIND_NATIVE_ARCH}, "
               "static runtime ${LITEMIND_STATIC_RUNTIME}")
