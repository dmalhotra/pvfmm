/**
 * \file profile.cpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 2-11-2011
 * \brief This file contains implementation of the class Profile.
 */

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <string>
#include <vector>
#include <stack>
#include <pvfmm_common.hpp>
#include <profile.hpp>

namespace pvfmm{

void Profile::Add_FLOP(long long inc){
#if __PROFILE__ >= 0
  if(!enable_state) return;
  #pragma omp critical (FLOP)
  FLOP+=inc;
#endif
}

void Profile::Add_MEM(long long inc){
#if __PROFILE__ >= 0
  if(!enable_state) return;
  #pragma omp critical (MEM)
  {
  MEM+=inc;
  for(size_t i=0;i<max_mem.size();i++)
    if(max_mem[i]<MEM) max_mem[i]=MEM;
  }
#endif
}

void Profile::Tic(const char* name_, const MPI_Comm* comm_,bool sync_, int verbose){
#if __PROFILE__ >= 0
  //sync_=true;
  if(!enable_state) return;
  if(verbose<=__PROFILE__ && verb_level.size()==enable_depth){
    if(comm_!=NULL && sync_) MPI_Barrier(*comm_);
    #ifdef __VERBOSE__
    int rank=0;
    if(comm_!=NULL) MPI_Comm_rank(*comm_,&rank);
    if(!rank){
      for(size_t i=0;i<name.size();i++) std::cout<<"    ";
      std::cout << "\033[1;31m"<<std::string(name_)<<"\033[0m {\n";
    }
    #endif
    name.push(std::string(name_));
    comm.push((MPI_Comm*)comm_);
    sync.push(sync_);
    max_mem.push_back(MEM);

    e_log.push_back(true);
    s_log.push_back(sync_);
    n_log.push_back(name.top());
    t_log.push_back(omp_get_wtime());
    f_log.push_back(FLOP);

    m_log.push_back(MEM);
    max_m_log.push_back(MEM);
    enable_depth++;
  }
  verb_level.push(verbose);
#endif
}

void Profile::Toc(){
#if __PROFILE__ >= 0
  if(!enable_state) return;
  ASSERT_WITH_MSG(!verb_level.empty(),"Unbalanced extra Toc()");
  if(verb_level.top()<=__PROFILE__ && verb_level.size()==enable_depth){
    ASSERT_WITH_MSG(!name.empty() && !comm.empty() && !sync.empty() && !max_mem.empty(),"Unbalanced extra Toc()");
    std::string name_=name.top();
    MPI_Comm* comm_=comm.top();
    bool sync_=sync.top();
    //sync_=true;

    e_log.push_back(false);
    s_log.push_back(sync_);
    n_log.push_back(name_);
    t_log.push_back(omp_get_wtime());
    f_log.push_back(FLOP);

    m_log.push_back(MEM);
    max_m_log.push_back(max_mem.back());

    #ifndef NDEBUG
    if(comm_!=NULL && sync_) MPI_Barrier(*comm_);
    #endif
    name.pop();
    comm.pop();
    sync.pop();
    max_mem.pop_back();

    #ifdef __VERBOSE__
    int rank=0;
    if(comm_!=NULL) MPI_Comm_rank(*comm_,&rank);
    if(!rank){
      for(size_t i=0;i<name.size();i++) std::cout<<"    ";
      std::cout<<"}\n";
    }
    #endif
    enable_depth--;
  }
  verb_level.pop();
#endif
}

void Profile::print(const MPI_Comm* comm_){
#if __PROFILE__ >= 0
  ASSERT_WITH_MSG(name.empty(),"Missing balancing Toc()");

  int np, rank;
  MPI_Comm c_self=MPI_COMM_SELF;
  if(comm_==NULL) comm_=&c_self;
  MPI_Barrier(*comm_);

  MPI_Comm_size(*comm_,&np);
  MPI_Comm_rank(*comm_,&rank);

  std::stack<double> tt;
  std::stack<long long> ff;
  std::stack<long long> mm;
  int width=10;
  size_t level=0;
  if(!rank && e_log.size()>0){
    std::cout<<"\n"<<std::setw(width*3-2*level)<<std::string(" ");
    std::cout<<"  "<<std::setw(width)<<std::string("t_min");
    std::cout<<"  "<<std::setw(width)<<std::string("t_avg");
    std::cout<<"  "<<std::setw(width)<<std::string("t_max");
    std::cout<<"  "<<std::setw(width)<<std::string("f_min");
    std::cout<<"  "<<std::setw(width)<<std::string("f_avg");
    std::cout<<"  "<<std::setw(width)<<std::string("f_max");

    std::cout<<"  "<<std::setw(width)<<std::string("f/s_min");
    std::cout<<"  "<<std::setw(width)<<std::string("f/s_max");
    std::cout<<"  "<<std::setw(width)<<std::string("f/s_total");

    std::cout<<"  "<<std::setw(width)<<std::string("m_init");
    std::cout<<"  "<<std::setw(width)<<std::string("m_max");
    std::cout<<"  "<<std::setw(width)<<std::string("m_final")<<'\n';
  }

  std::stack<std::string> out_stack;
  std::string s;
  out_stack.push(s);
  for(size_t i=0;i<e_log.size();i++){
    if(e_log[i]){
      level++;
      tt.push(t_log[i]);
      ff.push(f_log[i]);
      mm.push(m_log[i]);

      std::string s;
      out_stack.push(s);
    }else{
      double t0=t_log[i]-tt.top();tt.pop();
      double f0=(double)(f_log[i]-ff.top())*1e-9;ff.pop();
      double fs0=f0/t0;
      double t_max, t_min, t_sum, t_avg;
      double f_max, f_min, f_sum, f_avg;
      double fs_max, fs_min, fs_sum;//, fs_avg;
      double m_init, m_max, m_final;
      MPI_Reduce(&t0, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, *comm_);
      MPI_Reduce(&f0, &f_max, 1, MPI_DOUBLE, MPI_MAX, 0, *comm_);
      MPI_Reduce(&fs0, &fs_max, 1, MPI_DOUBLE, MPI_MAX, 0, *comm_);

      MPI_Reduce(&t0, &t_min, 1, MPI_DOUBLE, MPI_MIN, 0, *comm_);
      MPI_Reduce(&f0, &f_min, 1, MPI_DOUBLE, MPI_MIN, 0, *comm_);
      MPI_Reduce(&fs0, &fs_min, 1, MPI_DOUBLE, MPI_MIN, 0, *comm_);

      MPI_Reduce(&t0, &t_sum, 1, MPI_DOUBLE, MPI_SUM, 0, *comm_);
      MPI_Reduce(&f0, &f_sum, 1, MPI_DOUBLE, MPI_SUM, 0, *comm_);

      m_final=(double)m_log[i]*1e-9;
      m_init =(double)mm.top()*1e-9; mm.pop();
      m_max  =(double)max_m_log[i]*1e-9;

      t_avg=t_sum/np;
      f_avg=f_sum/np;
      //fs_avg=f_avg/t_max;
      fs_sum=f_sum/t_max;

      if(!rank){
        std::string s=out_stack.top();out_stack.pop();
        std::string s1=out_stack.top();out_stack.pop();
        std::stringstream ss(std::stringstream::in | std::stringstream::out);
        ss<<setiosflags(std::ios::fixed)<<std::setprecision(4)<<std::setiosflags(std::ios::left);

        for(size_t j=0;j<level-1;j++){
          size_t l=i+1;
          size_t k=level-1;
          while(k>j && l<e_log.size()){
            k+=(e_log[l]?1:-1);
            l++;
          }
          if(l<e_log.size()?e_log[l]:false)
            ss<<"| ";
          else
            ss<<"  ";
        }
        ss<<"+-";
        ss<<std::setw(width*3-2*level)<<n_log[i];
        ss<<std::setiosflags(std::ios::right);
        ss<<"  "<<std::setw(width)<<t_min;
        ss<<"  "<<std::setw(width)<<t_avg;
        ss<<"  "<<std::setw(width)<<t_max;

        ss<<"  "<<std::setw(width)<<f_min;
        ss<<"  "<<std::setw(width)<<f_avg;
        ss<<"  "<<std::setw(width)<<f_max;

        ss<<"  "<<std::setw(width)<<fs_min;
        //ss<<"  "<<std::setw(width)<<fs_avg;
        ss<<"  "<<std::setw(width)<<fs_max;
        ss<<"  "<<std::setw(width)<<fs_sum;

        ss<<"  "<<std::setw(width)<<m_init;
        ss<<"  "<<std::setw(width)<<m_max;
        ss<<"  "<<std::setw(width)<<m_final<<'\n';

        s1+=ss.str()+s;
        if(!s.empty() && (i+1<e_log.size()?e_log[i+1]:false)){
          for(size_t j=0;j<level;j++){
            size_t l=i+1;
            size_t k=level-1;
            while(k>j && l<e_log.size()){
              k+=(e_log[l]?1:-1);
              l++;
            }
            if(l<e_log.size()?e_log[l]:false)
              s1+=std::string("| ");
            else
              s1+=std::string("  ");
          }
          s1+=std::string("\n");
        }// */
        out_stack.push(s1);
      }
      level--;
    }
  }
  if(!rank)
    std::cout<<out_stack.top()<<'\n';

  reset();
#endif
}

void Profile::reset(){
  FLOP=0;
  while(!sync.empty())sync.pop();
  while(!name.empty())name.pop();
  while(!comm.empty())comm.pop();

  e_log.clear();
  s_log.clear();
  n_log.clear();
  t_log.clear();
  f_log.clear();
  m_log.clear();
  max_m_log.clear();
}

long long Profile::FLOP=0;
long long Profile::MEM=0;
bool Profile::enable_state=false;
std::stack<bool> Profile::sync;
std::stack<std::string> Profile::name;
std::stack<MPI_Comm*> Profile::comm;
std::vector<long long> Profile::max_mem;

unsigned int Profile::enable_depth=0;
std::stack<int> Profile::verb_level;

std::vector<bool> Profile::e_log;
std::vector<bool> Profile::s_log;
std::vector<std::string> Profile::n_log;
std::vector<double> Profile::t_log;
std::vector<long long> Profile::f_log;
std::vector<long long> Profile::m_log;
std::vector<long long> Profile::max_m_log;

}//end namespace
