#!/bin/bash
rm *.o
rm *.mod
rm *.exe

# icpx -O3 -I$CONDA_PREFIX/include -c reactor.cpp 
icpx -O3 -I$CONDA_PREFIX/include -I$CONDA_PREFIX/include/eigen3 -c reactor.cpp

# ifx -O3 \
#     -L$CONDA_PREFIX/lib \
#     -I$CONDA_PREFIX/include \
#     -o PaSR.exe \
#     PaSR.f90 reactor.o \
#     -lstdc++ -lcantera \
#     -Bstatic -L$HOME/code/isat/ISAT/lib -I$HOME/code/isat/ISAT/isatab_ser -lisat7_ser \
#     -qmkl -Bdynamic -qmkl \
#     -Wl,-rpath,$CONDA_PREFIX/lib
ifx -O3 \
    -I$CONDA_PREFIX/include \
    -I$HOME/code/isat/ISAT/isatab_ser \
    -o main.exe \
    PaSR.f90 reactor.o \
    $HOME/code/isat/ISAT/lib/libisatab_ser.a \
    $HOME/code/isat/ISAT/lib/libisat7_ser.a \
    $HOME/code/isat/ISAT/lib/libice_pic.a \
    $HOME/code/isat/ISAT/lib/libell.a \
    $HOME/code/isat/ISAT/lib/libsell.a \
    $HOME/code/isat/ISAT/lib/libceq.a \
    $HOME/code/isat/ISAT/lib/libck.a \
    $HOME/code/isat/ISAT/lib/libisatck.a \
    -L$CONDA_PREFIX/lib \
    -lstdc++ \
    -lcantera \
    -lsundials_cvodes \
    -lsundials_nvecserial \
    -lsundials_sunmatrixdense \
    -lsundials_sunlinsoldense \
    -lsundials_core \
    -qmkl \
    -Wl,-rpath,$CONDA_PREFIX/lib