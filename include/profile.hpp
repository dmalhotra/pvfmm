/**
 * \file profile.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains definition of the class Profile.
 */

#include <mpi.h>
#include <string>
#include <vector>
#include <stack>

#include <pvfmm_common.hpp>

#ifndef _PVFMM_PROFILE_HPP_
#define _PVFMM_PROFILE_HPP_

#ifndef __PROFILE__
#define __PROFILE__ -1
#endif

namespace pvfmm{

class Profile{
  public:

    static long long Add_FLOP(long long inc);

    static long long Add_MEM(long long inc);

    static bool Enable(bool state);

    static void Tic(const char* name_, const MPI_Comm* comm_=NULL,bool sync_=false, int level=0);

    static void Toc();

    static void print(const MPI_Comm* comm_=NULL);

    static void reset();
  private:

  static long long FLOP;
  static long long MEM;
  static bool enable_state;
  static std::stack<bool> sync;
  static std::stack<std::string> name;
  static std::stack<MPI_Comm*> comm;
  static std::vector<long long> max_mem;

  static unsigned int enable_depth;
  static std::stack<int> verb_level;

  static std::vector<bool> e_log;
  static std::vector<bool> s_log;
  static std::vector<std::string> n_log;
  static std::vector<double> t_log;
  static std::vector<long long> f_log;
  static std::vector<long long> m_log;
  static std::vector<long long> max_m_log;
};

}//end namespace

#endif //_PVFMM_PROFILE_HPP_
