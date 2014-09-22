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



MemoryManager::MemHead* MemoryManager::GetMemHead(void* p){
  static uintptr_t alignment=MEM_ALIGN-1;
  static uintptr_t header_size=(uintptr_t)(sizeof(MemoryManager::MemHead)+alignment) & ~(uintptr_t)alignment;
  return (MemHead*)(((char*)p)-header_size);
}

void* MemoryManager::malloc(const size_t& n_elem, const size_t& type_size, const uintptr_t& type_id) const{
  if(!n_elem) return NULL;
  static uintptr_t alignment=MEM_ALIGN-1;
  static uintptr_t header_size=(uintptr_t)(sizeof(MemHead)+alignment) & ~(uintptr_t)alignment;

  size_t size=n_elem*type_size+header_size;
  size=(uintptr_t)(size+alignment) & ~(uintptr_t)alignment;
  char* base=NULL;

  omp_set_lock(&omp_lock);
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
  omp_unset_lock(&omp_lock);
  if(!base){ // Use system malloc
    size+=2+alignment;
    char* p = (char*)::malloc(size);
    base = (char*)((uintptr_t)(p+2+alignment) & ~(uintptr_t)alignment);
    ((uint16_t*)base)[-1] = (uint16_t)(base-p);
  }

  { // Check out-of-bounds write
    #ifndef NDEBUG
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
    mem_head->type_id=type_id;
  }
  { // Set header check_sum
    #ifndef NDEBUG
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

void MemoryManager::free(void* p, const size_t& type_size, const uintptr_t& type_id) const{
  if(!p) return;
  static uintptr_t alignment=MEM_ALIGN-1;
  static uintptr_t header_size=(uintptr_t)(sizeof(MemHead)+alignment) & ~(uintptr_t)alignment;

  char* base=(char*)((char*)p-header_size);
  MemHead* mem_head=(MemHead*)base;
  assert(mem_head->type_id==type_id);

  if(base<&buff[0] || base>=&buff[buff_size]){ // Use system free
    char* p=(char*)((uintptr_t)base-((uint16_t*)base)[-1]);
    return ::free(p);
  }

  size_t n_indx=mem_head->n_indx;
  assert(n_indx>0 && n_indx<=node_buff.size());
  { // Verify header check_sum; set array to init_mem_val
    #ifndef NDEBUG
    { // Verify header check_sum
      size_t check_sum=0;
      for(size_t i=0;i<header_size;i++){
        check_sum+=base[i];
      }
      check_sum-=mem_head->check_sum;
      check_sum=check_sum & ((1UL << sizeof(mem_head->check_sum))-1);
      assert(check_sum==mem_head->check_sum);
    }
    size_t size=mem_head->n_elem*type_size;
    #pragma omp parallel for
    for(size_t i=0;i<size;i++) ((char*)p)[i]=init_mem_val;
    for(size_t i=0;i<sizeof(MemHead);i++) base[i]=init_mem_val;
    #endif
  }

  omp_set_lock(&omp_lock);
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
      size_t n_prev_indx=n.prev;
      MemNode& n_prev=node_buff[n_prev_indx-1];
      n_prev.next=n_indx;
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
      size_t n_next_indx=n.next;
      MemNode& n_next=node_buff[n_next_indx-1];
      n_next.prev=n_indx;
    }
  }
  n.free=true; // Insert n to free_map
  n.it=free_map.insert(std::make_pair(n.size,n_indx));
  omp_unset_lock(&omp_lock);
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
  T* A=(T*)mem_mgr->malloc(n_elem, sizeof(T), TypeTraits<T>::ID());

  if(!TypeTraits<T>::IsPOD()){ // Call constructors
    //printf("%s\n", __PRETTY_FUNCTION__);
    #pragma omp parallel for
    for(size_t i=0;i<n_elem;i++){
      T* Ai=new(A+i) T();
      assert(Ai==(A+i));
    }
  }else{
    for(size_t i=0;i<n_elem*sizeof(T);i++){
      ((char*)A)[i]=0;
    }
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
    size_t n_elem=mem_head->n_elem;
    for(size_t i=0;i<n_elem;i++){
      (A+i)->~T();
    }
  }else{
    MemoryManager::MemHead* mem_head=MemoryManager::GetMemHead(A);
    size_t n_elem=mem_head->n_elem;
    for(size_t i=0;i<n_elem*sizeof(T);i++){
      ((char*)A)[i]=0;
    }
  }

  static MemoryManager def_mem_mgr(0);
  if(!mem_mgr) mem_mgr=&def_mem_mgr;
  mem_mgr->free(A, sizeof(T), TypeTraits<T>::ID());
}

void* memcopy( void * destination, const void * source, size_t num){
  if(destination==source || num==0) return destination;
  return memcpy ( destination, source, num );
}

}//end namespace
}//end namespace

