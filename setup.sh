#!/bin/bash
pyopengl_platform="egl"
if [ ${pyopengl_platform} != ${PYOPENGL_PLATFORM} ]
then
    echo export PYOPENGL_PLATFORM=\"${pyopengl_platform}\" >> ~/.bashrc
fi