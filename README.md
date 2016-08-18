# aiformatter
C++ formatter based on RNN and clang

## Prerequisites
* Ubuntu
* Python3
 * **sudo apt-get install python3**
* NumPy
 * **sudo pip3 install numpy**
* Clang 3.8 + patched bindings included (may be 3.6 also supported)
 * **sudo apt-get install libclang-3.8-dev**
 * Configuring of aiformatter.py will be needed possibly to point clang.so location

## Usage

./ailearner.py <sample file, concatenation of many Mbs of code base> [APPEND]

This program takes existed code base on input and produces model.dat - learned RNN.

./aiformatter.py <source file> [model file]

Formatter takes source file and model and produces source file formatted according to model.
