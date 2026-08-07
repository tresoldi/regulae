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
    src/format.c
    src/loaders.c
    src/model.c
    src/multilect.c
    src/options.c
    src/scoring.c
    src/search.c
    src/syllabification.c
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

    add_executable(test_syllabification tests/c/test_syllabification.c)
    target_link_libraries(test_syllabification PRIVATE regulae::regulae)
    add_test(NAME syllabification COMMAND test_syllabification)

    add_executable(test_format tests/c/test_format.c)
    target_link_libraries(test_format PRIVATE regulae::regulae)
    add_test(NAME format COMMAND test_format)

    add_executable(test_loaders tests/c/test_loaders.c)
    target_compile_definitions(test_loaders PRIVATE REGULAE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(test_loaders PRIVATE regulae::regulae)
    add_test(NAME loaders COMMAND test_loaders)

    add_executable(test_scoring tests/c/test_scoring.c)
    target_link_libraries(test_scoring PRIVATE regulae::regulae)
    add_test(NAME scoring COMMAND test_scoring)

    add_executable(test_align tests/c/test_align.c)
    target_link_libraries(test_align PRIVATE regulae::regulae)
    add_test(NAME align COMMAND test_align)

    add_executable(test_pairwise_model tests/c/test_pairwise_model.c)
    target_link_libraries(test_pairwise_model PRIVATE regulae::regulae)
    add_test(NAME pairwise_model COMMAND test_pairwise_model)

    add_executable(test_multilect_model tests/c/test_multilect_model.c)
    target_link_libraries(test_multilect_model PRIVATE regulae::regulae)
    add_test(NAME multilect_model COMMAND test_multilect_model)

    add_executable(test_real_fixtures tests/c/test_real_fixtures.c)
    target_compile_definitions(test_real_fixtures PRIVATE REGULAE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(test_real_fixtures PRIVATE regulae::regulae)
    add_test(NAME real_fixtures COMMAND test_real_fixtures)

    if(REGULAE_BUILD_CLI)
        add_test(NAME cli_version COMMAND regulae_cli version)
        set_tests_properties(cli_version PROPERTIES PASS_REGULAR_EXPRESSION "regulae 0.1.0")
        add_test(NAME cli_help COMMAND regulae_cli help)
        set_tests_properties(cli_help PROPERTIES PASS_REGULAR_EXPRESSION "Usage: regulae")

        add_test(NAME cli_train_summary
            COMMAND regulae_cli train "${CMAKE_CURRENT_SOURCE_DIR}/testdata/parity/conditioned_multilect.tsv")
        set_tests_properties(cli_train_summary PROPERTIES
            PASS_REGULAR_EXPRESSION "LECTS	alpha beta gamma")

        add_test(NAME cli_train_human
            COMMAND regulae_cli train --human "${CMAKE_CURRENT_SOURCE_DIR}/testdata/parity/conditioned_multilect.tsv")
        set_tests_properties(cli_train_human PROPERTIES
            PASS_REGULAR_EXPRESSION "MultiLectModel")

        add_test(NAME cli_align
            COMMAND regulae_cli align "${CMAKE_CURRENT_SOURCE_DIR}/testdata/parity/three_lect_basic.tsv")
        set_tests_properties(cli_align PROPERTIES PASS_REGULAR_EXPRESSION "^ALIGN	c1")

        add_test(NAME cli_outliers
            COMMAND regulae_cli outliers --top-k 3 "${CMAKE_CURRENT_SOURCE_DIR}/testdata/parity/real_contaminated.tsv")
        set_tests_properties(cli_outliers PROPERTIES PASS_REGULAR_EXPRESSION "^OUTLIER	")

        add_test(NAME cli_wide
            COMMAND regulae_cli train --format wide "${CMAKE_CURRENT_SOURCE_DIR}/experiments/latin_spanish/cognates.tsv")
        set_tests_properties(cli_wide PROPERTIES PASS_REGULAR_EXPRESSION "LECTS	latin spanish")

        add_test(NAME cli_arcaverborum
            COMMAND regulae_cli train --format arcaverborum "${CMAKE_CURRENT_SOURCE_DIR}/testdata/parity/morph_boundary.csv")
        set_tests_properties(cli_arcaverborum PROPERTIES PASS_REGULAR_EXPRESSION "LECTS	one two")

        add_test(NAME cli_missing_file COMMAND regulae_cli train "${CMAKE_CURRENT_SOURCE_DIR}/no_such_corpus.tsv")
        set_tests_properties(cli_missing_file PROPERTIES WILL_FAIL TRUE)
    endif()
endif()
