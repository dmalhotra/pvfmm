/**
 * \file matrix.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains definition of the class Matrix.
 */

#include <stdint.h>
#include <cstdlib>

#include <pvfmm_common.hpp>
#include <vector.hpp>

#ifndef _PVFMM_MATRIX_HPP_
#define _PVFMM_MATRIX_HPP_

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif
namespace pvfmm{

template <class T>
class Permutation;

template <class T>
class Matrix{

  template <class Y>
  friend std::ostream& operator<<(std::ostream& output, const Matrix<Y>& M);

  public:

  Matrix();

  Matrix(size_t dim1, size_t dim2, sctl::Iterator<T> data_=sctl::NullIterator<T>(), bool own_data_=true);

#if defined(SCTL_MEMDEBUG)
  Matrix(size_t dim1, size_t dim2, T* data_, bool own_data_=true)
    : Matrix(dim1, dim2, (data_? sctl::Ptr2Itr<T>(data_, (sctl::Long)dim1*(sctl::Long)dim2) : sctl::NullIterator<T>()), own_data_) {}
#endif

  Matrix(const Matrix<T>& M);

  // See Vector<T>'s move ctor — same rationale.
  ~Matrix();

  void Swap(Matrix<T>& M);

  void ReInit(size_t dim1, size_t dim2, sctl::Iterator<T> data_=sctl::NullIterator<T>(), bool own_data_=true);

#if defined(SCTL_MEMDEBUG)
  void ReInit(size_t dim1, size_t dim2, T* data_, bool own_data_=true) {
    ReInit(dim1, dim2, (data_? sctl::Ptr2Itr<T>(data_, (sctl::Long)dim1*(sctl::Long)dim2) : sctl::NullIterator<T>()), own_data_);
  }
#endif

  void Write(const char* fname);

  void Read(const char* fname);

  size_t Dim(size_t i) const;

  void Resize(size_t i, size_t j);

  void SetZero();

  T* Begin();

  const T* Begin() const;

  Matrix<T>& operator=(const Matrix<T>& M);

  Matrix<T>& operator+=(const Matrix<T>& M);

  Matrix<T>& operator-=(const Matrix<T>& M);

  Matrix<T> operator+(const Matrix<T>& M2);

  Matrix<T> operator-(const Matrix<T>& M2);

  const T& operator()(size_t i,size_t j) const;

  T* operator[](size_t i);

  const T* operator[](size_t i) const;

  Matrix<T> operator*(const Matrix<T>& M);

  static void GEMM(Matrix<T>& M_r, const Matrix<T>& A, const Matrix<T>& B, T beta=0.0);

  // cublasgemm wrapper
  static void CUBLASGEMM(Matrix<T>& M_r, const Matrix<T>& A, const Matrix<T>& B, T beta=0.0);

  void RowPerm(const Permutation<T>& P);
  void ColPerm(const Permutation<T>& P);

  Matrix<T> Transpose() const;

  static void Transpose(Matrix<T>& M_r, const Matrix<T>& M);

  // Original matrix is destroyed.
  void SVD(Matrix<T>& tU, Matrix<T>& tS, Matrix<T>& tVT);

  // Original matrix is destroyed.
  Matrix<T> pinv(T eps=-1);

  private:

  size_t dim[2];
  sctl::Iterator<T> data_ptr;
  bool own_data;
};

template <class Y>
std::ostream& operator<<(std::ostream& output, const Matrix<Y>& M);


/**
 * /brief P=[e(p1)*s1 e(p2)*s2 ... e(pn)*sn],
 * where e(k) is the kth unit vector,
 * perm := [p1 p2 ... pn] is the permutation vector,
 * scal := [s1 s2 ... sn] is the scaling vector.
 */
#define PVFMM_PERM_INT_T size_t
template <class T>
class Permutation{

  template <class Y>
  friend std::ostream& operator<<(std::ostream& output, const Permutation<Y>& P);

  public:

  Permutation(){}

  Permutation(size_t size);

  static Permutation<T> RandPerm(size_t size);

  Matrix<T> GetMatrix() const;

  size_t Dim() const;

  Permutation<T> Transpose();

  Permutation<T>& operator*=(const Permutation<T>& P);

  Permutation<T> operator*(const Permutation<T>& P);

  Matrix<T> operator*(const Matrix<T>& M);

  template <class Y>
  friend Matrix<Y> operator*(const Matrix<Y>& M, const Permutation<Y>& P);

  Vector<PVFMM_PERM_INT_T> perm;
  Vector<T> scal;
};

template <class T>
Matrix<T> operator*(const Matrix<T>& M, const Permutation<T>& P);

/**
 * Transpose the in_dim1 x in_dim2 row-major matrix at `in` into `out`
 * (which receives in_dim2 x in_dim1). If in==out, the transpose is done in
 * place, staging the input through per-thread scratch storage; otherwise
 * the two ranges must not overlap.
 */
template <class T>
void MatrixTranspose(size_t in_dim1, size_t in_dim2, sctl::ConstIterator<T> in, sctl::Iterator<T> out);

#if defined(SCTL_MEMDEBUG)
// Legacy compatibility: accept raw pointers and wrap into iterators.
template <class T>
void MatrixTranspose(size_t in_dim1, size_t in_dim2, const T* in, T* out){
  const sctl::Long n=(sctl::Long)in_dim1*(sctl::Long)in_dim2;
  MatrixTranspose<T>(in_dim1, in_dim2,
      (in ? sctl::Ptr2ConstItr<T>(in, n) : sctl::ConstIterator<T>(sctl::NullIterator<T>())),
      (out? sctl::Ptr2Itr<T>(out, n) : sctl::NullIterator<T>()));
}
#endif

template <class Y>
std::ostream& operator<<(std::ostream& output, const Permutation<Y>& P);

}//end namespace
#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#include <matrix.txx>

#endif //_PVFMM_MATRIX_HPP_
