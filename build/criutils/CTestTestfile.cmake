# CMake generated Testfile for 
# Source directory: /home/karthik/ur5_ws/src/criutils
# Build directory: /home/karthik/ur5_ws/build/criutils
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_criutils_nosetests_tests "/home/karthik/ur5_ws/build/catkin_generated/env_cached.sh" "/usr/bin/python3" "/opt/ros/noetic/share/catkin/cmake/test/run_tests.py" "/home/karthik/ur5_ws/build/test_results/criutils/nosetests-tests.xml" "--return-code" "\"/home/karthik/.local/lib/python3.8/site-packages/cmake/data/bin/cmake\" -E make_directory /home/karthik/ur5_ws/build/test_results/criutils" "/usr/bin/nosetests3 -P --process-timeout=60 --where=/home/karthik/ur5_ws/src/criutils/tests --with-xunit --xunit-file=/home/karthik/ur5_ws/build/test_results/criutils/nosetests-tests.xml")
set_tests_properties(_ctest_criutils_nosetests_tests PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/noetic/share/catkin/cmake/test/tests.cmake;160;add_test;/opt/ros/noetic/share/catkin/cmake/test/nosetests.cmake;83;catkin_run_tests_target;/home/karthik/ur5_ws/src/criutils/CMakeLists.txt;12;catkin_add_nosetests;/home/karthik/ur5_ws/src/criutils/CMakeLists.txt;0;")
