#include <mpi.h>
#include <complex>

#ifndef __PVFMM_DTYPES_H_
#define __PVFMM_DTYPES_H_

/**
 * \file	dtypes.h
 * \brief	Traits to determine MPI_DATATYPE from a C++ datatype
 * \author	Hari Sundar, hsundar@gmail.com

  Traits to determine MPI_DATATYPE from a C++ datatype. For non standard
  C++ datatypes (like classes), we will need to define additional classes. An
  example is given for the case of the std. complex variable. Additional
  classes can be added as required.
 */

namespace pvfmm{
namespace par{

  /**
   * \class Mpi_datatype
   * \brief An abstract class used for communicating messages using user-defined
   * datatypes. The user must implement the static member function "value()" that
   * returns the MPI_Datatype corresponding to this user-defined datatype.
   * \author Hari Sundar, hsundar@gmail.com
   * \see Mpi_datatype<bool>
   */
  template <typename T>
  class Mpi_datatype{
   public:
    static MPI_Datatype value() {
      static bool first = true;
      static MPI_Datatype datatype;
      if (first) {
        first = false;
        MPI_Type_contiguous(sizeof(T), MPI_BYTE, &datatype);
        MPI_Type_commit(&datatype);
      }
      return datatype;
    }

    static MPI_Op sum() {
      static bool   first = true;
      static MPI_Op myop;

      if (first) {
        first = false;
        int commune=1;
        MPI_Op_create(sum_fn, commune, &myop);
      }

      return myop;
    }

    static MPI_Op max() {
      static bool   first = true;
      static MPI_Op myop;

      if (first) {
        first = false;
        int commune=1;
        MPI_Op_create(max_fn, commune, &myop);
      }

      return myop;
    }

   private:

    static void sum_fn( void * a_, void * b_, int * len_, MPI_Datatype * datatype){
      T* a=(T*)a_;
      T* b=(T*)b_;
      int len=*len_;
      for(int i=0;i<len;i++){
        b[i]=a[i]+b[i];
      }
    }

    static void max_fn( void * a_, void * b_, int * len_, MPI_Datatype * datatype){
      T* a=(T*)a_;
      T* b=(T*)b_;
      int len=*len_;
      for(int i=0;i<len;i++){
        if(a[i]>b[i]) b[i]=a[i];
      }
    }

  };

  #define HS_MPIDATATYPE(CTYPE, MPITYPE) \
    template <> \
    class Mpi_datatype<CTYPE> { \
     public: \
      static MPI_Datatype value() { \
        return MPITYPE; \
      } \
      static MPI_Op sum() { \
        return MPI_SUM; \
      } \
      static MPI_Op max() { \
        return MPI_MAX; \
      } \
    };

  HS_MPIDATATYPE(short,          MPI_SHORT)
  HS_MPIDATATYPE(int,            MPI_INT)
  HS_MPIDATATYPE(long,           MPI_LONG)
  HS_MPIDATATYPE(unsigned short, MPI_UNSIGNED_SHORT)
  HS_MPIDATATYPE(unsigned int,   MPI_UNSIGNED)
  HS_MPIDATATYPE(unsigned long,  MPI_UNSIGNED_LONG)
  HS_MPIDATATYPE(float,          MPI_FLOAT)
  HS_MPIDATATYPE(double,         MPI_DOUBLE)
  HS_MPIDATATYPE(long double,    MPI_LONG_DOUBLE)
  HS_MPIDATATYPE(long long,      MPI_LONG_LONG_INT)
  HS_MPIDATATYPE(char,           MPI_CHAR)
  HS_MPIDATATYPE(unsigned char,  MPI_UNSIGNED_CHAR)

  //PetscScalar is simply a typedef for double. Hence no need to explicitly
  //define an mpi_datatype for it.

  #undef HS_MPIDATATYPE

  template <typename T>
  class Mpi_datatype<std::complex<T> > {
   public:
    static MPI_Datatype value() {
      static bool         first = true;
      static MPI_Datatype datatype;

      if (first) {
        first = false;
        MPI_Type_contiguous(2, Mpi_datatype<T>::value(), &datatype);
        MPI_Type_commit(&datatype);
      }

      return datatype;
    }
  };

} //end namespace
} //end namespace

#endif //__PVFMM_DTYPES_H_
