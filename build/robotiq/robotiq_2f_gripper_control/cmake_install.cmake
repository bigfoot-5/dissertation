# Install script for directory: /home/karthik/ur5_ws/src/robotiq/robotiq_2f_gripper_control

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/karthik/ur5_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/robotiq_2f_gripper_control/msg" TYPE FILE FILES
    "/home/karthik/ur5_ws/src/robotiq/robotiq_2f_gripper_control/msg/Robotiq2FGripper_robot_input.msg"
    "/home/karthik/ur5_ws/src/robotiq/robotiq_2f_gripper_control/msg/Robotiq2FGripper_robot_output.msg"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/karthik/ur5_ws/build/robotiq/robotiq_2f_gripper_control/catkin_generated/safe_execute_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/robotiq_2f_gripper_control/cmake" TYPE FILE FILES "/home/karthik/ur5_ws/build/robotiq/robotiq_2f_gripper_control/catkin_generated/installspace/robotiq_2f_gripper_control-msg-paths.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/karthik/ur5_ws/devel/include/robotiq_2f_gripper_control")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/karthik/ur5_ws/devel/share/roseus/ros/robotiq_2f_gripper_control")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/karthik/ur5_ws/devel/share/common-lisp/ros/robotiq_2f_gripper_control")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/karthik/ur5_ws/devel/share/gennodejs/ros/robotiq_2f_gripper_control")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3" -m compileall "/home/karthik/ur5_ws/devel/lib/python3/dist-packages/robotiq_2f_gripper_control")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/karthik/ur5_ws/devel/lib/python3/dist-packages/robotiq_2f_gripper_control" REGEX "/\\_\\_init\\_\\_\\.py$" EXCLUDE REGEX "/\\_\\_init\\_\\_\\.pyc$" EXCLUDE)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/karthik/ur5_ws/devel/lib/python3/dist-packages/robotiq_2f_gripper_control" FILES_MATCHING REGEX "/home/karthik/ur5_ws/devel/lib/python3/dist-packages/robotiq_2f_gripper_control/.+/__init__.pyc?$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/karthik/ur5_ws/build/robotiq/robotiq_2f_gripper_control/catkin_generated/installspace/robotiq_2f_gripper_control.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/robotiq_2f_gripper_control/cmake" TYPE FILE FILES "/home/karthik/ur5_ws/build/robotiq/robotiq_2f_gripper_control/catkin_generated/installspace/robotiq_2f_gripper_control-msg-extras.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/robotiq_2f_gripper_control/cmake" TYPE FILE FILES
    "/home/karthik/ur5_ws/build/robotiq/robotiq_2f_gripper_control/catkin_generated/installspace/robotiq_2f_gripper_controlConfig.cmake"
    "/home/karthik/ur5_ws/build/robotiq/robotiq_2f_gripper_control/catkin_generated/installspace/robotiq_2f_gripper_controlConfig-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/robotiq_2f_gripper_control" TYPE FILE FILES "/home/karthik/ur5_ws/src/robotiq/robotiq_2f_gripper_control/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/robotiq_2f_gripper_control" TYPE DIRECTORY FILES "/home/karthik/ur5_ws/src/robotiq/robotiq_2f_gripper_control/include/robotiq_2f_gripper_control/" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/robotiq_2f_gripper_control/robotiq_2f_gripper_ethercat_node" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/robotiq_2f_gripper_control/robotiq_2f_gripper_ethercat_node")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/robotiq_2f_gripper_control/robotiq_2f_gripper_ethercat_node"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/robotiq_2f_gripper_control" TYPE EXECUTABLE FILES "/home/karthik/ur5_ws/devel/lib/robotiq_2f_gripper_control/robotiq_2f_gripper_ethercat_node")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/robotiq_2f_gripper_control/robotiq_2f_gripper_ethercat_node" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/robotiq_2f_gripper_control/robotiq_2f_gripper_ethercat_node")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/robotiq_2f_gripper_control/robotiq_2f_gripper_ethercat_node"
         OLD_RPATH "/opt/ros/noetic/lib:/home/karthik/ur5_ws/devel/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/robotiq_2f_gripper_control/robotiq_2f_gripper_ethercat_node")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/robotiq_2f_gripper_control" TYPE PROGRAM FILES
    "/home/karthik/ur5_ws/src/robotiq/robotiq_2f_gripper_control/nodes/Robotiq2FGripperRtuNode.py"
    "/home/karthik/ur5_ws/src/robotiq/robotiq_2f_gripper_control/nodes/Robotiq2FGripperSimpleController.py"
    "/home/karthik/ur5_ws/src/robotiq/robotiq_2f_gripper_control/nodes/Robotiq2FGripperStatusListener.py"
    "/home/karthik/ur5_ws/src/robotiq/robotiq_2f_gripper_control/nodes/Robotiq2FGripperTcpNode.py"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/robotiq_2f_gripper_control" TYPE PROGRAM FILES "/home/karthik/ur5_ws/src/robotiq/robotiq_2f_gripper_control/src/robotiq_2f_gripper_control/robotiq_2f_gripper_ctrl.py")
endif()

