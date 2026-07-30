cmake_minimum_required(VERSION 3.20)

project(regulae
    VERSION 0.1.0
    DESCRIPTION "Sound-correspondence rule discovery"
    LANGUAGES C
)

option(REGULAE_BUILD_TESTS "Build native C tests" ON)
option(REGULAE_BUILD_CLI "Build the regulae CLI" ON)
option(REGULAE_BUILD_PYTHON_TESTS "Reserve Python wrapper test option" ON)
set(REGULAE_ENABLE_SANITIZER "" CACHE STRING "Sanitizer: address or undefined")
set_property(CACHE REGULAE_ENABLE_SANITIZER PROPERTY STRINGS "" address undefined)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(REGULAE_ENABLE_SANITIZER)
    if(MSVC)
        message(FATAL_ERROR "REGULAE_ENABLE_SANITIZER is not supported with MSVC yet")
    endif()
    if(NOT REGULAE_ENABLE_SANITIZER MATCHES "^(address|undefined)$")
        message(FATAL_ERROR "REGULAE_ENABLE_SANITIZER must be empty, address, or undefined")
    endif()
endif()

find_package(merkmal CONFIG QUIET)
if(NOT TARGET merkmal::merkmal)
    set(REGULAE_MERKMAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../merkmal" CACHE PATH "Merkmal C source checkout")
    if(EXISTS "${REGULAE_MERKMAL_SOURCE_DIR}/CMakeLists.txt")
        set(MERKMAL_BUILD_TESTS OFF CACHE BOOL "Build merkmal tests" FORCE)
        add_subdirectory("${REGULAE_MERKMAL_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/_deps/merkmal")
    else()
        message(FATAL_ERROR "merkmal::merkmal not found; install merkmal or set REGULAE_MERKMAL_SOURCE_DIR")
    endif()
endif()

add_library(regulae
    src/context.c
    src/memory.c
    src/model.c
    src/options.c
    src/scoring.c
    src/search.c
    src/types.c
    src/version.c
)
add_library(regulae::regulae ALIAS regulae)
target_compile_features(regulae PUBLIC c_std_99)
target_include_directories(regulae PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(regulae PUBLIC merkmal::merkmal)
if(UNIX AND NOT APPLE)
    target_link_libraries(regulae PUBLIC m)
endif()
set_target_properties(regulae PROPERTIES C_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
target_compile_definitions(regulae PRIVATE REGULAE_BUILDING_LIBRARY)
if(BUILD_SHARED_LIBS)
    target_compile_definitions(regulae PUBLIC REGULAE_SHARED)
endif()

if(MSVC)
    target_compile_options(regulae PRIVATE /W4)
else()
    target_compile_options(regulae PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

if(REGULAE_ENABLE_SANITIZER)
    target_compile_options(regulae PUBLIC -fsanitize=${REGULAE_ENABLE_SANITIZER} -fno-omit-frame-pointer)
    target_link_options(regulae PUBLIC -fsanitize=${REGULAE_ENABLE_SANITIZER})
endif()

if(REGULAE_BUILD_CLI)
    add_executable(regulae_cli cmd/regulae-c/main.c)
    set_target_properties(regulae_cli PROPERTIES OUTPUT_NAME regulae)
    target_link_libraries(regulae_cli PRIVATE regulae::regulae)
endif()

include(GNUInstallDirs)
install(TARGETS regulae
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(REGULAE_BUILD_CLI)
    install(TARGETS regulae_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(FILES include/regulae.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(REGULAE_BUILD_TESTS)
    include(CTest)

    add_executable(test_options tests/c/test_options.c)
    target_link_libraries(test_options PRIVATE regulae::regulae)
    add_test(NAME options COMMAND test_options)

    add_executable(test_context tests/c/test_context.c)
    target_link_libraries(test_context PRIVATE regulae::regulae)
    add_test(NAME context COMMAND test_context)

    add_executable(test_merkmal_bridge tests/c/test_merkmal_bridge.c)
    target_link_libraries(test_merkmal_bridge PRIVATE regulae::regulae)
    add_test(NAME merkmal_bridge COMMAND test_merkmal_bridge)

    add_executable(test_scoring tests/c/test_scoring.c)
    target_link_libraries(test_scoring PRIVATE regulae::regulae)
    add_test(NAME scoring COMMAND test_scoring)

    add_executable(test_align tests/c/test_align.c)
    target_link_libraries(test_align PRIVATE regulae::regulae)
    add_test(NAME align COMMAND test_align)

    add_executable(test_pairwise_model tests/c/test_pairwise_model.c)
    target_link_libraries(test_pairwise_model PRIVATE regulae::regulae)
    add_test(NAME pairwise_model COMMAND test_pairwise_model)

    add_executable(test_real_fixtures tests/c/test_real_fixtures.c)
    target_compile_definitions(test_real_fixtures PRIVATE REGULAE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(test_real_fixtures PRIVATE regulae::regulae)
    add_test(NAME real_fixtures COMMAND test_real_fixtures)

    if(REGULAE_BUILD_CLI)
        add_test(NAME cli_version COMMAND regulae_cli version)
        set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "regulae 0.1.0")
        add_test(NAME cli_help COMMAND regulae_cli help)
        set_tests_properties(cli_help PROPERTIES PASS_REGULAR_EXPRESSION "Usage: regulae")
    endif()
endif()
