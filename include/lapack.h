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

#ifndef _LAPACK_H_
#define _LAPACK_H_

//EXTERN_C_BEGIN
extern "C"
{
  extern void sgesvd_(char *JOBU, char *JOBVT, int *M, int *N, float *A, int *LDA,
                                                        float *S, float *U, int *LDU, float *VT, int *LDVT, float *WORK, int *LWORK, int *INFO);
  /*!    DGESVD computes the singular value decomposition (SVD) of a real
        *  M-by-N matrix A, optionally computing the left and/or right singular
        *  vectors. The SVD is written
        *
        *       A = U * SIGMA * transpose(V)
        *
        *  where SIGMA is an M-by-N matrix which is zero except for its
        *  min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
        *  V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
        *  are the singular values of A; they are real and non-negative, and
        *  are returned in descending order.  The first min(m,n) columns of
        *  U and V are the left and right singular vectors of A.
        *
        * See http://www.netlib.org/lapack/double/dgesvd.f for more information
        */
  extern void dgesvd_(char *JOBU, char *JOBVT, int *M, int *N, double *A, int *LDA,
                                                        double *S, double *U, int *LDU, double *VT, int *LDVT, double *WORK, int *LWORK, int *INFO);
  /*! DGESDD computes the singular value decomposition (SVD) of a real
        *  M-by-N matrix A, optionally computing the left and right singular
        *  vectors.  If singular vectors are desired, it uses a
        * divide-and-conquer algorithm.
        *
        *  The SVD is written
        *
        *       A = U * SIGMA * transpose(V)
        *
        *  where SIGMA is an M-by-N matrix which is zero except for its
        *  min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
        *  V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
`       *  are the singular values of A; they are real and non-negative, and
        *  are returned in descending order.  The first min(m,n) columns of
        *  U and V are the left and right singular vectors of A.
        *
        *  See http://www.netlib.org/lapack/double/dgesdd.f for more information
        */
  extern void dgesdd_(char *jobz, int* m, int* n, double* a, int* lda,
                                                        double* s, double* u, int* ldu, double* vt, int* ldvt, double* work, int* lwork, int* iwork, int* info);
  /*!  DGETRF computes an LU factorization of a general M-by-N matrix A
        *  using partial pivoting with row interchanges.
        *
        *  The factorization has the form
        *
        *            A = P * L * U
        *
        *  where P is a permutation matrix, L is lower triangular with unit
        *  diagonal elements (lower trapezoidal if m > n), and U is upper
        *  triangular (upper trapezoidal if m < n).
        *
        *  See http://www.netlib.org/lapack/double/dgetrf.f for more information
        */
  extern void dgetrf_(int *M, int *N, double *A, int *LDA, int *IPIV, int *INFO);
  /*!  DGETRI computes the inverse of a matrix using the LU factorization
        *  computed by DGETRF.
        *
        *  This method inverts U and then computes inv(A) by solving the system
        *  inv(A)*L = inv(U) for inv(A).
        *
        *  See http://www.netlib.org/lapack/double/dgetri.f for more information
        */
  extern void dgetri_(int *N, double *A, int *LDA, int *IPIV, double *WORK, int *LWORK, int *INFO);
}
//EXTERN_C_END

#endif
