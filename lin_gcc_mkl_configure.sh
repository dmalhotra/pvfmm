
unset PVFMM_DIR
./configure MPICXX=mpicxx --prefix=$HOME/local \
--with-openmp-flag='fopenmp' \
CXXFLAGS="-O3 -march=native -std=c++14 -DPVFMM_FFTW3_MKL" \
--with-fftw-include="$MKLROOT/include/fftw" \
--with-fftw-lib="-lmkl_rt"
