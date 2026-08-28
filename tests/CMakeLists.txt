# Each test links the shared engine library, so nothing is compiled twice and a
# test can never drift from the build the executable uses.

function(litemind_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE litemind_core)
    # Tests must be as self-contained as the executable, or they fail to start
    # for reasons that have nothing to do with what they are testing.
    if(LITEMIND_STATIC_RUNTIME AND MSVC)
        set_property(TARGET ${name} PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
    add_test(NAME ${name} COMMAND ${name} ${ARGN})
endfunction()

# These need no model files and run everywhere.
litemind_add_test(JsonTest)
litemind_add_test(SafeTensorTest)
litemind_add_test(KernelTest)
litemind_add_test(RopeTest)
litemind_add_test(MoeRouterSamplerTest)
litemind_add_test(ExpertCacheTest)
litemind_add_test(AttentionTest)

# These read the real tokenizer, so they are registered only when it is present.
if(EXISTS "${CMAKE_SOURCE_DIR}/models/tokenizer.json")
    litemind_add_test(TokenizerRoundTrip "${CMAKE_SOURCE_DIR}/models/tokenizer.json")
else()
    add_executable(TokenizerRoundTrip TokenizerRoundTrip.cpp)
    target_link_libraries(TokenizerRoundTrip PRIVATE litemind_core)
    if(LITEMIND_STATIC_RUNTIME AND MSVC)
        set_property(TARGET TokenizerRoundTrip PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
    message(STATUS "TokenizerRoundTrip built but not registered: models/tokenizer.json is absent")
endif()
