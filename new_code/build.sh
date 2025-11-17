export CANTERA_INCLUDE_DIR=/mnt/c/Aug2025/canteraInstall/include
export CANTERA_LIB_DIR=/mnt/c/Aug2025/canteraInstall/lib
export ISATAB_SERIAL_DIR=/mnt/c/complete/ISAT/isatab_ser
export ISAT_LIB_DIR=/mnt/c/complete/ISAT/lib
export CVODES_INC=/mnt/c/Aug2025/sundialsInstall/include
export CVODES_LIB=/mnt/c/Aug2025/sundialsInstall/lib
export REACTOR_LIB=/mnt/c/complete/fromScratch_Apr12/newHENN/finalFinal
rm *.o
rm *.mod
rm *.exe
icpx -O3 -L$CONDA_PREFIX/lib -I$CONDA_PREFIX/include -I/usr/include/eigen3 -c reactor.cpp -std=c++17
ifx -O3 -L$CONDA_PREFIX/lib -I$CONDA_PREFIX/include -I/usr/include/eigen3 \
	-o PaSR.exe PaSR.f90 reactor.o -std=c++17 -lstdc++ \
	-lsundials_cvodes -lsundials_ida -lsundials_idas -lsundials_nvecserial \
	-lcantera -Wl,-rpath,$CONDA_PREFIX/lib \
	-Bstatic -L$ISAT_LIB_DIR -I$ISATAB_SERIAL_DIR -lisat7_ser \
	-mkl -Bdynamic -mkl -lstdc++
