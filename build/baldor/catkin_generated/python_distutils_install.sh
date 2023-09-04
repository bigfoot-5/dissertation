#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/karthik/ur5_ws/src/baldor"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/karthik/ur5_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/karthik/ur5_ws/install/lib/python3/dist-packages:/home/karthik/ur5_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/karthik/ur5_ws/build" \
    "/usr/bin/python3" \
    "/home/karthik/ur5_ws/src/baldor/setup.py" \
     \
    build --build-base "/home/karthik/ur5_ws/build/baldor" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/karthik/ur5_ws/install" --install-scripts="/home/karthik/ur5_ws/install/bin"
