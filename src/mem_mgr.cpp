/**
 * \file mem_mgr.cpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 9-21-2014
 * \brief This file contains the definition of a simple memory manager which
 * uses a pre-allocated buffer of size defined in call to the constructor.
 */

#include <mem_mgr.hpp>

#include <omp.h>
#include <iostream>
#include <cassert>

namespace pvfmm{
namespace mem{

MemoryManager::MemoryManager(size_t N){
  buff_size=N;
  { // Allocate buff
    assert(MEM_ALIGN <= 0x8000);
    size_t alignment=MEM_ALIGN-1;
    char* base_ptr=(char*)::malloc(N+2+alignment); assert(base_ptr);
    buff=(char*)((uintptr_t)(base_ptr+2+alignment) & ~(uintptr_t)alignment);
    ((uint16_t*)buff)[-1] = (uint16_t)(buff-base_ptr);
  }
  { // Initialize to init_mem_val
    #ifndef NDEBUG
    #pragma omp parallel for
    for(size_t i=0;i<buff_size;i++){
      buff[i]=init_mem_val;
    }
    #endif
  }
  n_dummy_indx=new_node();
  size_t n_indx=new_node();
  MemNode& n_dummy=node_buff[n_dummy_indx-1];
  MemNode& n=node_buff[n_indx-1];

  n_dummy.size=0;
  n_dummy.free=false;
  n_dummy.prev=0;
  n_dummy.next=n_indx;
  n_dummy.mem_ptr=&buff[0];
  assert(n_indx);

  n.size=N;
  n.free=true;
  n.prev=n_dummy_indx;
  n.next=0;
  n.mem_ptr=&buff[0];
  n.it=free_map.insert(std::make_pair(N,n_indx));

  omp_init_lock(&omp_lock);
}

MemoryManager::~MemoryManager(){
  MemNode* n_dummy=&node_buff[n_dummy_indx-1];
  MemNode* n=&node_buff[n_dummy->next-1];
  if(!n->free || n->size!=buff_size ||
      node_stack.size()!=node_buff.size()-2){
    std::cout<<"\nWarning: memory leak detected.\n";
  }
  omp_destroy_lock(&omp_lock);

  { // Check out-of-bounds write
    #ifndef NDEBUG
    #pragma omp parallel for
    for(size_t i=0;i<buff_size;i++){
      assert(buff[i]==init_mem_val);
    }
    #endif
  }
  { // free buff
    assert(buff);
    ::free(buff-((uint16_t*)buff)[-1]);
  }
}

void MemoryManager::print() const{
  if(!buff_size) return;
  omp_set_lock(&omp_lock);

  size_t size=0;
  size_t largest_size=0;
  MemNode* n=&node_buff[n_dummy_indx-1];
  std::cout<<"\n|";
  while(n->next){
    n=&node_buff[n->next-1];
    if(n->free){
      std::cout<<' ';
      largest_size=std::max(largest_size,n->size);
    }
    else{
      std::cout<<'#';
      size+=n->size;
    }
  }
  std::cout<<"|  allocated="<<round(size*1000.0/buff_size)/10<<"%";
  std::cout<<"  largest_free="<<round(largest_size*1000.0/buff_size)/10<<"%\n";

  omp_unset_lock(&omp_lock);
}

void MemoryManager::test(){
  size_t M=2000000000;
  { // With memory manager
    size_t N=M*sizeof(double)*1.1;
    double tt;
    double* tmp;

    std::cout<<"With memory manager: ";
    MemoryManager memgr(N);

    for(size_t j=0;j<3;j++){
      tmp=(double*)memgr.malloc(M*sizeof(double)); assert(tmp);
      tt=omp_get_wtime();
      #pragma omp parallel for
      for(size_t i=0;i<M;i+=64) tmp[i]=i;
      tt=omp_get_wtime()-tt;
      std::cout<<tt<<' ';
      memgr.free(tmp);
    }
    std::cout<<'\n';
  }
  { // Without memory manager
    double tt;
    double* tmp;

    std::cout<<"Without memory manager: ";
    for(size_t j=0;j<3;j++){
      tmp=(double*)::malloc(M*sizeof(double)); assert(tmp);
      tt=omp_get_wtime();
      #pragma omp parallel for
      for(size_t i=0;i<M;i+=64) tmp[i]=i;
      tt=omp_get_wtime()-tt;
      std::cout<<tt<<' ';
      ::free(tmp);
    }
    std::cout<<'\n';
  }
}

MemoryManager glbMemMgr(GLOBAL_MEM_BUFF*1024LL*1024LL);

}//end namespace
}//end namespace

