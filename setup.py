# Please don't change the order of following packages!
import sys
from distutils.core import setup

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, "python version >= 3.6 is required"

setup(
    name="policydissect",
    install_requires=[
        "metadrive-simulator~=0.2.5.1",
        "keyboard",
        "yapf",
    ],
)
