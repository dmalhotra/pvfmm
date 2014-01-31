#ifndef __DTYPES_H_
#define __DTYPES_H_

#include <mpi.h>
#include <complex>

/**
 * \file	dtypes.h
 * \brief	Traits to determine MPI_DATATYPE from a C++ datatype
 * \author	Hari Sundar, hsundar@gmail.com

  Traits to determine MPI_DATATYPE from a C++ datatype. For non standard
  C++ datatypes (like classes), we will need to define additional classes. An
  example is given for the case of the std. complex variable. Additional
  classes can be added as required.
 */

namespace par {

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
  };

  #define HS_MPIDATATYPE(CTYPE, MPITYPE) \
    template <> \
    class Mpi_datatype<CTYPE> { \
     public: \
      static MPI_Datatype value() { \
        return MPITYPE; \
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

#endif

