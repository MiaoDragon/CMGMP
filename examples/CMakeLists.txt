find_package(mujoco REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(toppra REQUIRED)

if (mujoco_FOUND)
    message("find mujoco")
else()
    message("cannot find mujoco")
endif()
link_libraries(mujoco::mujoco Eigen3::Eigen toppra)
set(MJ_INCLUDE_DIR "$ENV{MJ_PATH}/include")



set(dart_libs dart dart-utils-urdf dart-gui)

set(example_required_libraries 
    cmgmp-utils cmgmp-contact cmgmp-world cmgmp-search 
    ${dart_libs}
    "${mujoco_LIBRARIES}"
    "glfw"
    )

set(example_include_dirs 
    "${PROJECT_SOURCE_DIR}" 
    "${PROJECT_SOURCE_DIR}/cmgmp/external/eigen3" 
    "${PROJECT_SOURCE_DIR}/cmgmp/external/dartsim"
    "${MJ_INCLUDE_DIR}"
    )

set(example_link_dirs 
    "${PROJECT_BINARY_DIR}" 
    "${PROJECT_BINARY_DIR}/cmgmp/external/dartsim"
    )




include_directories(${example_include_dirs})
link_libraries(${example_required_libraries})

add_executable(kr5 KR5.cpp)
add_executable(push_along_walls push_along_walls.cpp)
add_executable(ddhand_bookself ddhand_bookshelf.cpp)
add_executable(ddhand_flip ddhand_flip.cpp)
add_executable(ddhand_slideup ddhand_slideup.cpp)
add_executable(peg_out_of_hole peg_out_hole.cpp)
add_executable(bookshelf bookshelf.cpp)
add_executable(pick_card pick_card.cpp)
add_executable(flip_cube flip_cube.cpp)
add_executable(book_push book_push.cpp)
add_executable(mujoco_pushing mujoco_pushing.cpp)
add_executable(mujoco_pushing_point_vis mujoco_pushing_point_vis.cpp)
add_executable(mujoco_pushing_pt mujoco_pushing_pt.cpp)
add_executable(mujoco_push_pt_iterative mujoco_push_pt_iterative.cpp)
add_executable(mujoco_push_pt_iterative_teleport mujoco_push_pt_iterative_teleport.cpp)