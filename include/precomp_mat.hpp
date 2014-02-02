/**
 * \file precomp_mat.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 3-07-2011
 * \brief This file contains the definition of the PrecompMat class.
 * Handles storage of precomputed translation matrices.
 */

#ifndef _PVFMM_PrecompMAT_HPP_
#define _PVFMM_PrecompMAT_HPP_

#include <mpi.h>
#include <pvfmm_common.hpp>
#include <matrix.hpp>

namespace pvfmm{

typedef enum{
  UC2UE_Type= 0,
  DC2DE_Type= 1,
  S2U_Type  = 2,
  U2U_Type  = 3,
  D2D_Type  = 4,
  D2T_Type  = 5,
  U0_Type   = 6,
  U1_Type   = 7,
  U2_Type   = 8,
  V_Type    = 9,
  W_Type    =10,
  X_Type    =11,
  V1_Type   =12,
  BC_Type   =13,
  Type_Count=14
} Mat_Type;

typedef enum{
  ReflecX = 0,
  ReflecY = 1,
  ReflecZ = 2,
  SwapXY  = 3,
  SwapXZ  = 4,
  R_Perm = 0,
  C_Perm = 5,
  Perm_Count=10
} Perm_Type;

template <class T>
class PrecompMat{

 public:

  PrecompMat(bool homogen, int max_d);

  Matrix<T>& Mat(int l, Mat_Type type, size_t indx);

  Permutation<T>& Perm_R(Mat_Type type, size_t indx);

  Permutation<T>& Perm_C(Mat_Type type, size_t indx);

  Permutation<T>& Perm(Mat_Type type, size_t indx);

  size_t CompactData(int l, Mat_Type type, Matrix<char>& comp_data, size_t offset=0);

  void Save2File(const char* fname, bool replace=false);

  void LoadFile(const char* fname, MPI_Comm comm);

  std::vector<T>& RelativeTrgCoord();

  bool Homogen();

 private:

  std::vector<std::vector<Matrix     <T> > > mat;
  std::vector<std::vector<Permutation<T> > > perm;
  std::vector<std::vector<Permutation<T> > > perm_r;
  std::vector<std::vector<Permutation<T> > > perm_c;
  std::vector<T> rel_trg_coord;

  bool homogeneous;
  int max_depth;
};

}//end namespace

#include <precomp_mat.txx>

#endif //_PrecompMAT_HPP_
