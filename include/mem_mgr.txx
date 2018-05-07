/**
 * \file mem_mgr.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 9-21-2014
 * \brief This file contains the definition of a simple memory manager which
 * uses a pre-allocated buffer of size defined in call to the constructor.
 */

#include <omp.h>
#include <algorithm>
#include <cstring>
#include <cassert>
#include <device_wrapper.hpp>

namespace pvfmm{
namespace mem{

template <class T>
uintptr_t TypeTraits<T>::ID(){
  return (uintptr_t)&ID;
}

template <class T>
bool TypeTraits<T>::IsPOD(){
  return false;
}

#define PVFMMDefinePOD(type) template<> bool inline TypeTraits<type>::IsPOD(){return true;};
PVFMMDefinePOD(char);
PVFMMDefinePOD(float);
PVFMMDefinePOD(double);
PVFMMDefinePOD(int);
PVFMMDefinePOD(long long);
PVFMMDefinePOD(unsigned long);
PVFMMDefinePOD(char*);
PVFMMDefinePOD(float*);
PVFMMDefinePOD(double*);
#undef PVFMMDefinePOD


MemoryManager::MemHead* MemoryManager::GetMemHead(void* p){
  static constexpr uintptr_t alignment=MEM_ALIGN-1;
  static constexpr uintptr_t header_size=(uintptr_t)(sizeof(MemoryManager::MemHead)+alignment) & ~(uintptr_t)alignment;
  return (MemHead*)(((char*)p)-header_size);
}

size_t MemoryManager::new_node() const{
  if(node_stack.empty()){
    node_buff.resize(node_buff.size()+1);
    node_stack.push(node_buff.size());
  }

  size_t indx=node_stack.top();
  node_stack.pop();
  assert(indx);
  return indx;
}

void MemoryManager::delete_node(size_t indx) const{
  assert(indx);
  assert(indx<=node_buff.size());
  MemNode& n=node_buff[indx-1];
  n.free=false;
  n.size=0;
  n.prev=0;
  n.next=0;
  n.mem_ptr=NULL;
  node_stack.push(indx);
}


template <class T>
T* aligned_new(size_t n_elem, const MemoryManager* mem_mgr){
  if(!n_elem) return NULL;

  static MemoryManager def_mem_mgr(0);
  if(!mem_mgr) mem_mgr=&def_mem_mgr;
  T* A=(T*)mem_mgr->malloc(n_elem, sizeof(T));

  if(!TypeTraits<T>::IsPOD()){ // Call constructors
    //printf("%s\n", __PRETTY_FUNCTION__);
    #pragma omp parallel for
    for(size_t i=0;i<n_elem;i++){
      T* Ai=new(A+i) T();
      assert(Ai==(A+i));
    }
  }else{
    #ifndef NDEBUG
    #pragma omp parallel for
    for(size_t i=0;i<n_elem*sizeof(T);i++){
      ((char*)A)[i]=0;
    }
    #endif
  }

  assert(A);
  return A;
}

template <class T>
void aligned_delete(T* A, const MemoryManager* mem_mgr){
  if (!A) return;

  if(!TypeTraits<T>::IsPOD()){ // Call destructors
    //printf("%s\n", __PRETTY_FUNCTION__);
    MemoryManager::MemHead* mem_head=MemoryManager::GetMemHead(A);
    size_t type_size=mem_head->type_size;
    size_t n_elem=mem_head->n_elem;
    for(size_t i=0;i<n_elem;i++){
      ((T*)(((char*)A)+i*type_size))->~T();
    }
  }else{
    #ifndef NDEBUG
    MemoryManager::MemHead* mem_head=MemoryManager::GetMemHead(A);
    size_t type_size=mem_head->type_size;
    size_t n_elem=mem_head->n_elem;
    size_t size=n_elem*type_size;
    #pragma omp parallel for
    for(size_t i=0;i<size;i++){
      ((char*)A)[i]=0;
    }
    #endif
  }

  static MemoryManager def_mem_mgr(0);
  if(!mem_mgr) mem_mgr=&def_mem_mgr;
  mem_mgr->free(A);
}

template <class ValueType> ValueType* copy(ValueType* destination, const ValueType* source, size_t num){
  if (destination != source && num) {
    if (TypeTraits<ValueType>::IsPOD()) {
      memcpy(destination, source, num * sizeof(ValueType));
    } else {
      for (size_t i = 0; i < num; i++) destination[i] = source[i];
    }
  }
  return destination;
}

}//end namespace
}//end namespace

