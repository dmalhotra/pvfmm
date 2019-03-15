/**
 * \file mem_mgr.cpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 9-21-2014
 * \brief This file contains the definition of a simple memory manager which
 * uses a pre-allocated buffer of size defined in call to the constructor.
 */

#include <mpi.h>
#include <mem_mgr.hpp>

#include <omp.h>
#include <iostream>
#include <cassert>
#include <cmath>

namespace pvfmm{
  PeriodicType periodicType=PeriodicType::NONE;
namespace mem{

MemoryManager::MemoryManager(size_t N){
  buff_size=N;
  { // Allocate buff
    assert(PVFMM_MEM_ALIGN <= 0x8000);
    size_t alignment=PVFMM_MEM_ALIGN-1;
    char* base_ptr=(char*)DeviceWrapper::host_malloc(N+2+alignment); assert(base_ptr);
    buff=(char*)((uintptr_t)(base_ptr+2+alignment) & ~(uintptr_t)alignment);
    ((uint16_t*)buff)[-1] = (uint16_t)(buff-base_ptr);
  }
  { // Initialize to init_mem_val
    #ifndef PVFMM_NDEBUG
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
}

MemoryManager::~MemoryManager(){
  MemNode* n_dummy=&node_buff[n_dummy_indx-1];
  MemNode* n=&node_buff[n_dummy->next-1];
  if(!n->free || n->size!=buff_size ||
      node_stack.size()!=node_buff.size()-2){
    std::cout<<"\nWarning: memory leak detected.\n";
  }

  { // Check out-of-bounds write
    #ifndef PVFMM_NDEBUG
    #pragma omp parallel for
    for(size_t i=0;i<buff_size;i++){
      assert(buff[i]==init_mem_val);
    }
    #endif
  }
  { // free buff
    assert(buff);
    DeviceWrapper::host_free(buff-((uint16_t*)buff)[-1]);
  }
}

void* MemoryManager::malloc(const size_t n_elem, const size_t type_size) const{
  if(!n_elem) return NULL;
  static constexpr uintptr_t alignment=PVFMM_MEM_ALIGN-1;
  static constexpr uintptr_t header_size=(uintptr_t)(sizeof(MemHead)+alignment) & ~(uintptr_t)alignment;

  size_t size=n_elem*type_size+header_size;
  size=(uintptr_t)(size+alignment) & ~(uintptr_t)alignment;
  char* base=NULL;

  mutex_lock.lock();
  std::multimap<size_t, size_t>::iterator it=free_map.lower_bound(size);
  size_t n_indx=(it!=free_map.end()?it->second:0);
  if(n_indx){ // Allocate from buff
    size_t n_free_indx=(it->first>size?new_node():0);
    MemNode& n=node_buff[n_indx-1];
    assert(n.size==it->first);
    assert(n.it==it);
    assert(n.free);

    if(n_free_indx){ // Create a node for the remaining free part.
      MemNode& n_free=node_buff[n_free_indx-1];
      n_free=n;
      n_free.size-=size;
      n_free.mem_ptr=(char*)n_free.mem_ptr+size;
      { // Insert n_free to the link list
        n_free.prev=n_indx;
        if(n_free.next){
          size_t n_next_indx=n_free.next;
          MemNode& n_next=node_buff[n_next_indx-1];
          n_next.prev=n_free_indx;
        }
        n.next=n_free_indx;
      }
      assert(n_free.free); // Insert n_free to free map
      n_free.it=free_map.insert(std::make_pair(n_free.size,n_free_indx));
      n.size=size; // Update n
    }

    n.free=false;
    free_map.erase(it);
    base = n.mem_ptr;
  }
  mutex_lock.unlock();
  if(!base){ // Use system malloc
    size+=2+alignment;
    char* p = (char*)DeviceWrapper::host_malloc(size);
    base = (char*)((uintptr_t)(p+2+alignment) & ~(uintptr_t)alignment);
    ((uint16_t*)base)[-1] = (uint16_t)(base-p);
  }

  { // Check out-of-bounds write
    #ifndef PVFMM_NDEBUG
    if(n_indx){
      #pragma omp parallel for
      for(size_t i=0;i<size;i++) assert(base[i]==init_mem_val);
    }
    #endif
  }

  MemHead* mem_head=(MemHead*)base;
  { // Set mem_head
    mem_head->n_indx=n_indx;
    mem_head->n_elem=n_elem;
    mem_head->type_size=type_size;
  }
  { // Set header check_sum
    #ifndef PVFMM_NDEBUG
    size_t check_sum=0;
    mem_head->check_sum=0;
    for(size_t i=0;i<header_size;i++){
      check_sum+=base[i];
    }
    check_sum=check_sum & ((1UL << sizeof(mem_head->check_sum))-1);
    mem_head->check_sum=check_sum;
    #endif
  }
  return (void*)(base+header_size);
}

void MemoryManager::free(void* p) const{
  if(!p) return;
  static constexpr uintptr_t alignment=PVFMM_MEM_ALIGN-1;
  static constexpr uintptr_t header_size=(uintptr_t)(sizeof(MemHead)+alignment) & ~(uintptr_t)alignment;

  char* base=(char*)((char*)p-header_size);
  MemHead* mem_head=(MemHead*)base;

  if(base<&buff[0] || base>=&buff[buff_size]){ // Use system free
    char* p_=(char*)((uintptr_t)base-((uint16_t*)base)[-1]);
    return DeviceWrapper::host_free(p_);
  }

  size_t n_indx=mem_head->n_indx;
  assert(n_indx>0 && n_indx<=node_buff.size());
  { // Verify header check_sum; set array to init_mem_val
    #ifndef PVFMM_NDEBUG
    { // Verify header check_sum
      size_t check_sum=0;
      for(size_t i=0;i<header_size;i++){
        check_sum+=base[i];
      }
      check_sum-=mem_head->check_sum;
      check_sum=check_sum & ((1UL << sizeof(mem_head->check_sum))-1);
      assert(check_sum==mem_head->check_sum);
    }
    size_t size=mem_head->n_elem*mem_head->type_size;
    #pragma omp parallel for
    for(size_t i=0;i<size;i++) ((char*)p)[i]=init_mem_val;
    for(size_t i=0;i<sizeof(MemHead);i++) base[i]=init_mem_val;
    #endif
  }

  mutex_lock.lock();
  MemNode& n=node_buff[n_indx-1];
  assert(!n.free && n.size>0 && n.mem_ptr==base);
  if(n.prev!=0 && node_buff[n.prev-1].free){
    size_t n_prev_indx=n.prev;
    MemNode& n_prev=node_buff[n_prev_indx-1];
    n.size+=n_prev.size;
    n.mem_ptr=n_prev.mem_ptr;
    n.prev=n_prev.prev;
    free_map.erase(n_prev.it);
    delete_node(n_prev_indx);

    if(n.prev){
      node_buff[n.prev-1].next=n_indx;
    }
  }
  if(n.next!=0 && node_buff[n.next-1].free){
    size_t n_next_indx=n.next;
    MemNode& n_next=node_buff[n_next_indx-1];
    n.size+=n_next.size;
    n.next=n_next.next;
    free_map.erase(n_next.it);
    delete_node(n_next_indx);

    if(n.next){
      node_buff[n.next-1].prev=n_indx;
    }
  }
  n.free=true; // Insert n to free_map
  n.it=free_map.insert(std::make_pair(n.size,n_indx));
  mutex_lock.unlock();
}

void MemoryManager::print() const{
  if(!buff_size) return;
  mutex_lock.lock();

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

  mutex_lock.unlock();
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
      for(size_t i=0;i<M;i+=64) tmp[i]=(double)i;
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
      tmp=(double*)DeviceWrapper::host_malloc(M*sizeof(double)); assert(tmp);
      tt=omp_get_wtime();
      #pragma omp parallel for
      for(size_t i=0;i<M;i+=64) tmp[i]=(double)i;
      tt=omp_get_wtime()-tt;
      std::cout<<tt<<' ';
      DeviceWrapper::host_free(tmp);
    }
    std::cout<<'\n';
  }
}

void MemoryManager::Check() const{
  #ifndef PVFMM_NDEBUG
  //print();
  mutex_lock.lock();
  MemNode* curr_node=&node_buff[n_dummy_indx-1];
  while(curr_node->next){
    if(curr_node->free){
      char* base=curr_node->mem_ptr;
      #pragma omp parallel for
      for(size_t i=0;i<curr_node->size;i++){
        assert(base[i]==init_mem_val);
      }
    }
    curr_node=&node_buff[curr_node->next-1];
  }
  mutex_lock.unlock();
  #endif
}

MemoryManager glbMemMgr(PVFMM_GLOBAL_MEM_BUFF*1024LL*1024LL);

}//end namespace
}//end namespace

