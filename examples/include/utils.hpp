/**
 * \file utils.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 1-1-2011
 */

#ifndef _UTILS_
#define _UTILS_

#include <vector>
#include <mpi.h>
#include <cheb_utils.hpp>
#include <fmm_tree.hpp>

template <class FMM_Mat_t>
void CheckFMMOutput(pvfmm::FMM_Tree<FMM_Mat_t>* mytree, const pvfmm::Kernel<typename FMM_Mat_t::Real_t>* mykernel);

template <class Real_t>
struct TestFn{
  typedef void (*Fn_t)(Real_t* c, int n, Real_t* out);
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
