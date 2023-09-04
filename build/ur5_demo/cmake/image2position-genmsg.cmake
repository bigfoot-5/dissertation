# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "image2position: 1 messages, 0 services")

set(MSG_I_FLAGS "-Iimage2position:/home/karthik/ur5_ws/src/ur5_demo/msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(image2position_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/karthik/ur5_ws/src/ur5_demo/msg/Array.msg" NAME_WE)
add_custom_target(_image2position_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "image2position" "/home/karthik/ur5_ws/src/ur5_demo/msg/Array.msg" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(image2position
  "/home/karthik/ur5_ws/src/ur5_demo/msg/Array.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/image2position
)

### Generating Services

### Generating Module File
_generate_module_cpp(image2position
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/image2position
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(image2position_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(image2position_generate_messages image2position_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/karthik/ur5_ws/src/ur5_demo/msg/Array.msg" NAME_WE)
add_dependencies(image2position_generate_messages_cpp _image2position_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(image2position_gencpp)
add_dependencies(image2position_gencpp image2position_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS image2position_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(image2position
  "/home/karthik/ur5_ws/src/ur5_demo/msg/Array.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/image2position
)

### Generating Services

### Generating Module File
_generate_module_eus(image2position
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/image2position
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(image2position_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(image2position_generate_messages image2position_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/karthik/ur5_ws/src/ur5_demo/msg/Array.msg" NAME_WE)
add_dependencies(image2position_generate_messages_eus _image2position_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(image2position_geneus)
add_dependencies(image2position_geneus image2position_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS image2position_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(image2position
  "/home/karthik/ur5_ws/src/ur5_demo/msg/Array.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/image2position
)

### Generating Services

### Generating Module File
_generate_module_lisp(image2position
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/image2position
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(image2position_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(image2position_generate_messages image2position_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/karthik/ur5_ws/src/ur5_demo/msg/Array.msg" NAME_WE)
add_dependencies(image2position_generate_messages_lisp _image2position_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(image2position_genlisp)
add_dependencies(image2position_genlisp image2position_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS image2position_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(image2position
  "/home/karthik/ur5_ws/src/ur5_demo/msg/Array.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/image2position
)

### Generating Services

### Generating Module File
_generate_module_nodejs(image2position
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/image2position
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(image2position_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(image2position_generate_messages image2position_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/karthik/ur5_ws/src/ur5_demo/msg/Array.msg" NAME_WE)
add_dependencies(image2position_generate_messages_nodejs _image2position_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(image2position_gennodejs)
add_dependencies(image2position_gennodejs image2position_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS image2position_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(image2position
  "/home/karthik/ur5_ws/src/ur5_demo/msg/Array.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/image2position
)

### Generating Services

### Generating Module File
_generate_module_py(image2position
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/image2position
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(image2position_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(image2position_generate_messages image2position_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/karthik/ur5_ws/src/ur5_demo/msg/Array.msg" NAME_WE)
add_dependencies(image2position_generate_messages_py _image2position_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(image2position_genpy)
add_dependencies(image2position_genpy image2position_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS image2position_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/image2position)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/image2position
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(image2position_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/image2position)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/image2position
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(image2position_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/image2position)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/image2position
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(image2position_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/image2position)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/image2position
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(image2position_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/image2position)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/image2position\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/image2position
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(image2position_generate_messages_py std_msgs_generate_messages_py)
endif()
