/**
  @file parUtils.h
  @brief A set of parallel utilities.
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  @author Hari Sundar, hsundar@gmail.com
  @author Shravan Veerapaneni, shravan@seas.upenn.edu
  @author Santi Swaroop Adavani, santis@gmail.com
  @author Dhairya Malhotra, dhairya.malhotra@gmail.com
  */

#include <mpi.h>
#include <vector>
#include <vector.hpp>

#ifndef __PVFMM_PAR_UTILS_H_
#define __PVFMM_PAR_UTILS_H_

/**
  @namespace par
  @author Rahul Sampath
  @author Hari Sundar
  @brief Collection of Generic Parallel Functions: Sorting, Partitioning, Searching,...
  */
namespace pvfmm{
namespace par{

  /**
    @author Rahul S. Sampath
    */
  template <typename T>
    int Mpi_Alltoallv_sparse(T* sendbuf, int* sendcnts, int* sdispls,
        T* recvbuf, int* recvcnts, int* rdispls, const MPI_Comm& comm);

  /**
    @author Rahul S. Sampath
    */
  template <typename T>
    int Mpi_Alltoallv_dense(T* sendbuf, int* sendcnts, int* sdispls,
        T* recvbuf, int* recvcnts, int* rdispls, const MPI_Comm& comm);

  /**
    @brief A parallel weighted partitioning function. In our implementation, we
    do not pose any restriction on the input or the number of processors. This
    function can be used with an odd number of processors as well.  Some
    processors can pass an empty vector as input. The relative ordering of the
    elements is preserved.
    @author Hari Sundar
    @author Rahul Sampath
    @param[in,out] vec       the input vector
    @param[in]     getWeight function pointer to compute the weight of each
    element. If you pass NULL, then every element will get a weight equal to 1.
    @param[in]     comm      the communicator
    */
  template<typename T>
    int partitionW(Vector<T>& vec,
        long long* wts, const MPI_Comm& comm);
  template<typename T>
    int partitionW(std::vector<T>& vec,
        long long* wts, const MPI_Comm& comm);

  /**
    @brief A parallel hyper quick sort implementation.
    @author Dhairya Malhotra
    @param[in]  in   the input vector
    @param[out] out  the output vector
    @param[in]  comm the communicator
    */
  template<typename T>
    int HyperQuickSort(const Vector<T>& in, Vector<T> & out, const MPI_Comm& comm);
  template<typename T>
    int HyperQuickSort(const std::vector<T>& in, std::vector<T> & out, const MPI_Comm& comm);

  template<typename A, typename B>
    struct SortPair{
      int operator<(const SortPair<A,B>& p1) const{ return key<p1.key;}

      A key;
      B data;
    };

  /**
    @brief Returns the scatter mapping which will sort the keys.
    @author Dhairya Malhotra
    @param[in]  key           the input keys to sort
    @param[out] scatter_index the output index vector for the scatter mapping
    @param[in]  comm          the MPI communicator
    @param[in]  split_key     for partitioning of sorted array, optional
    */
  template<typename T>
    int SortScatterIndex(const Vector<T>& key, Vector<size_t>& scatter_index,
        const MPI_Comm& comm, const T* split_key=NULL);

  /**
    @brief Forward scatter data based on scatter index.
    @author Dhairya Malhotra
    @param[in,out] data          the data to scatter
    @param[in]     scatter_index the index vector for the scatter mapping
    @param[in]     comm          the MPI communicator
    */
  template<typename T>
    int ScatterForward(Vector<T>& data, const Vector<size_t>& scatter_index,
        const MPI_Comm& comm);

  /**
    @brief Reverse scatter data based on scatter index.
    @author Dhairya Malhotra
    @param[in,out] data          the data to scatter
    @param[in]     scatter_index the index vector for the scatter mapping
    @param[in]     comm          the MPI communicator
    @param[in]     loc_size      the local array size after scatter, optional
    */
  template<typename T>
    int ScatterReverse(Vector<T>& data, const Vector<size_t>& scatter_index,
        const MPI_Comm& comm, size_t loc_size=0);

}//end namespace
}//end namespace

#include "parUtils.txx"

#endif //__PVFMM_PAR_UTILS_H_
