cmake_minimum_required(VERSION 3.16)
project(deformable_body_simulation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen3: try the usual CMake package config first (apt's libeigen3-dev
# installs one); fall back to pkg-config, which also works for the apt
# package on some distros.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(EIGEN3 REQUIRED eigen3)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIRS}"
    )
endif()

enable_testing()

# --- Main binary: one executable, dispatched by subcommand (see src/main.cpp) ---
add_executable(deformable_sim
    src/main.cpp
)
target_include_directories(deformable_sim PRIVATE include)
target_link_libraries(deformable_sim PRIVATE Eigen3::Eigen)
set_target_properties(deformable_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# --- Tests --------------------------------------------------------------------
add_executable(test_gradient_hessian
    tests/test_gradient_hessian.cpp
)
target_include_directories(test_gradient_hessian PRIVATE include)
target_link_libraries(test_gradient_hessian PRIVATE Eigen3::Eigen)
add_test(NAME gradient_hessian_check COMMAND test_gradient_hessian)

add_executable(test_xpbd_equilibrium
    tests/test_xpbd_equilibrium.cpp
)
target_include_directories(test_xpbd_equilibrium PRIVATE include)
target_link_libraries(test_xpbd_equilibrium PRIVATE Eigen3::Eigen)
add_test(NAME xpbd_equilibrium_check COMMAND test_xpbd_equilibrium)

# material_point_method/ is pure Python (Taichi) and has no CMake target --
# see the root README for how to run it.
