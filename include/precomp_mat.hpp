/**
 * \file precomp_mat.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 3-07-2011
 * \brief This file contains the definition of the PrecompMat class.
 * Handles storage of precomputed translation matrices.
 */

#include <mpi.h>
#include <vector>
#include <cstdlib>

#include <pvfmm_common.hpp>
#include <matrix.hpp>

#ifndef _PVFMM_PrecompMAT_HPP_
#define _PVFMM_PrecompMAT_HPP_

namespace pvfmm{

typedef enum{
  UC2UE0_Type= 0,
  UC2UE1_Type= 1,
  DC2DE0_Type= 2,
  DC2DE1_Type= 3,
  S2U_Type  = 4,
  U2U_Type  = 5,
  D2D_Type  = 6,
  D2T_Type  = 7,
  U0_Type   = 8,
  U1_Type   = 9,
  U2_Type   =10,
  V_Type    =11,
  W_Type    =12,
  X_Type    =13,
  V1_Type   =14,
  BC_Type   =15,
  Type_Count=16
} Mat_Type;

typedef enum{
  Scaling = 0,
  ReflecX = 1,
  ReflecY = 2,
  ReflecZ = 3,
  SwapXY  = 4,
  SwapXZ  = 5,
  R_Perm = 0,
  C_Perm = 6,
  Perm_Count=12
} Perm_Type;

template <class T>
class PrecompMat{

 public:

  PrecompMat(bool scale_invar);

  Matrix<T>& Mat(int l, Mat_Type type, size_t indx);

  Permutation<T>& Perm_R(int l, Mat_Type type, size_t indx);

  Permutation<T>& Perm_C(int l, Mat_Type type, size_t indx);

  Permutation<T>& Perm(Mat_Type type, size_t indx);

  size_t CompactData(int l, Mat_Type type, Matrix<char>& comp_data, size_t offset=0);

  void Save2File(const char* fname, bool replace=false);

  void LoadFile(const char* fname, MPI_Comm comm);

  std::vector<T>& RelativeTrgCoord();

  bool ScaleInvar();

 private:

  std::vector<std::vector<Matrix     <T> > > mat;
  std::vector<std::vector<Permutation<T> > > perm;
  std::vector<std::vector<Permutation<T> > > perm_r;
  std::vector<std::vector<Permutation<T> > > perm_c;
  std::vector<T> rel_trg_coord;

  bool scale_invar;
};

}//end namespace

#include <precomp_mat.txx>

#endif //_PrecompMAT_HPP_
