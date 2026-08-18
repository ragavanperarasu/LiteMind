add_executable(TokenizerRoundTrip
    TokenizerRoundTrip.cpp
    ${CMAKE_SOURCE_DIR}/src/Tokenizer.cpp
)

target_include_directories(TokenizerRoundTrip PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(TokenizerRoundTrip PRIVATE /W4 /permissive-)
else()
    target_compile_options(TokenizerRoundTrip PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(
    NAME TokenizerRoundTrip
    COMMAND TokenizerRoundTrip ${CMAKE_SOURCE_DIR}/models/tokenizer.json
)

add_executable(WeightReaderTest
    WeightReaderTest.cpp
    ${CMAKE_SOURCE_DIR}/src/Tensor.cpp
    ${CMAKE_SOURCE_DIR}/src/SafeTensor.cpp
    ${CMAKE_SOURCE_DIR}/src/WeightReader.cpp
)

target_include_directories(WeightReaderTest PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(WeightReaderTest PRIVATE /W4 /permissive-)
else()
    target_compile_options(WeightReaderTest PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME WeightReaderTest COMMAND WeightReaderTest)

add_executable(CpuTensorTest
    CpuTensorTest.cpp
    ${CMAKE_SOURCE_DIR}/src/CpuTensor.cpp
)

target_include_directories(CpuTensorTest PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(CpuTensorTest PRIVATE /W4 /permissive-)
else()
    target_compile_options(CpuTensorTest PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME CpuTensorTest COMMAND CpuTensorTest)

add_executable(MoeRouterSamplerTest
    MoeRouterSamplerTest.cpp
    ${CMAKE_SOURCE_DIR}/src/CpuTensor.cpp
    ${CMAKE_SOURCE_DIR}/src/MoeRouter.cpp
    ${CMAKE_SOURCE_DIR}/src/Sampler.cpp
)

target_include_directories(MoeRouterSamplerTest PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(MoeRouterSamplerTest PRIVATE /W4 /permissive-)
else()
    target_compile_options(MoeRouterSamplerTest PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME MoeRouterSamplerTest COMMAND MoeRouterSamplerTest)

add_executable(AttentionTest
    AttentionTest.cpp
    ${CMAKE_SOURCE_DIR}/src/KvCache.cpp
    ${CMAKE_SOURCE_DIR}/src/Attention.cpp
)

target_include_directories(AttentionTest PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(AttentionTest PRIVATE /W4 /permissive-)
else()
    target_compile_options(AttentionTest PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME AttentionTest COMMAND AttentionTest)
