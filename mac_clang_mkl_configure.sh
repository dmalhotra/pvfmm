unset PVFMM_DIR

export CXXFLAGS="-std=c++14 -O3 -march=native -DPVFMM_FFTW3_MKL" 
# embed rpath
export LDFLAGS="-L${MKLROOT}/lib -Wl,-rpath,${MKLROOT}/lib -lmkl_rt -L/opt/intel/lib -Wl,-rpath,/opt/intel/lib -liomp5 "

./configure --prefix=/Users/wyan/local/ \
--with-openmp-flag="fopenmp=libiomp5" \
--with-fftw-include="${MKLROOT}/include/fftw" \
--with-fftw-lib="-lmkl_rt" 

