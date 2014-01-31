/* Kernel Independent Fast Multipole Method
   Copyright (C) 2004 Lexing Ying, New York University

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with this program; see the file COPYING.  If not, write to the Free
Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
02111-1307, USA.  */

#ifndef _BLAS_H_
#define _BLAS_H_

extern "C"
{
  /*! DAXPY compute y := alpha * x + y where alpha is a scalar and x and y are n-vectors.
	*  See http://www.netlib.org/blas/daxpy.f for more information.
	*/
  void saxpy_(int* N, float* ALPHA, float* X, int* INCX, float* Y, int* INCY);
  void daxpy_(int* N, double* ALPHA, double* X, int* INCX, double* Y, int* INCY);
  /*!  DGEMM  performs one of the matrix-matrix operations
	*
	*     C := alpha*op( A )*op( B ) + beta*C,
	*
	*  where  op( X ) is one of
	*
	*     op( X ) = X   or   op( X ) = X',
	*
	*  alpha and beta are scalars, and A, B and C are matrices, with op( A )
	*  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
	*  See http://www.netlib.org/blas/dgemm.f for more information.
	*/
  void sgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, float* ALPHA, float* A,
				 int* LDA, float* B, int* LDB, float* BETA, float* C, int* LDC);
  void dgemm_(char* TRANSA, char* TRANSB, int* M, int* N, int* K, double* ALPHA, double* A,
				 int* LDA, double* B, int* LDB, double* BETA, double* C, int* LDC);
  /*!  DGEMV  performs one of the matrix-vector operations
	*
	*     y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
	*
	*  where alpha and beta are scalars, x and y are vectors and A is an m by n matrix.
	*  See http://www.netlib.org/blas/dgemv.f for more information
	*/
  void sgemv_(char* TRANS, int* M, int* N, float* ALPHA, float* A, int* LDA, float* X, int* INCX,
				 float* BETA, float* Y, int* INCY);
  void dgemv_(char* TRANS, int* M, int* N, double* ALPHA, double* A, int* LDA, double* X, int* INCX,
				 double* BETA, double* Y, int* INCY);
  /*!  DGER   performs the rank 1 operation
	*
	*     A := alpha*x*y' + A,
	*
	*  where alpha is a scalar, x is an m element vector, y is an n element
	*  vector and A is an m by n matrix.
	*  See http://www.netlib.org/blas/dger.f for more information
	*/
  void sger_ (int* M, int * N, float* ALPHA, float* X, int* INCX, float* Y, int* INCY,
				 float* A, int* LDA);
  void dger_ (int* M, int * N, double* ALPHA, double* X, int* INCX, double* Y, int* INCY,
				 double* A, int* LDA);
  /*! DSCAL computes y := alpha * y where alpha is a scalar and y is an n-vector.
	*  See http://www.netlib.org/blas/dscal.f for more information
	*/
  void sscal_(int* N, float* ALPHA, float* X, int* INCX);
  void dscal_(int* N, double* ALPHA, double* X, int* INCX);
}

#endif

