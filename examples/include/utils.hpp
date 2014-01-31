/**
 * \file utils.hpp
 * \author Dhairya Malhotra, dhairya.malhotra88@gmail.com
 * \date 1-1-2011
 */

#ifndef _UTILS_
#define _UTILS_

#include <vector>
#include <mpi.h>
#include <cheb_utils.hpp>
#include <fmm_tree.hpp>

template <class FMM_Mat_t>
void CheckFMMOutput(FMM_Tree<FMM_Mat_t>* mytree, Kernel<typename FMM_Mat_t::Real_t>* mykernel);

template <class Real_t>
struct TestFn;

template <>
struct TestFn<double>{
  typedef void (*Fn_t)(double* c, int n, double* out);
};

template <>
struct TestFn<float>{
  typedef void (*Fn_t)(float* c, int n, float* out);
};

template <class FMMTree_t>
void CheckChebOutput(FMMTree_t* mytree, typename TestFn<typename FMMTree_t::Real_t>::Fn_t fn_poten, int fn_dof, std::string t_name="");

enum DistribType{
  UnifGrid,
  RandUnif,
  RandGaus,
  RandElps,
  RandSphr
};

template <class Real_t>
std::vector<Real_t> point_distrib(DistribType, size_t N, MPI_Comm comm);

void commandline_option_start(int argc, char** argv, const char* help_text=NULL);

const char* commandline_option(int argc, char** argv, const char* opt, const char* def_val, bool required, const char* err_msg);

void commandline_option_end(int argc, char** argv);

#include <utils.txx>

#endif
