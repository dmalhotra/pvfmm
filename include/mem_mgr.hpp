/**
 * \file mem_mgr.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 6-30-2014
 * \brief This file contains the declaration of a simple memory manager which
 * uses a pre-allocated buffer of size defined in call to the constructor.
 */

// TODO: Implement fast stack allocation.

#include <omp.h>
#include <cstdlib>
#include <stdint.h>
#include <cassert>
#include <vector>
#include <stack>
#include <map>

#include <pvfmm_common.hpp>

#ifndef _PVFMM_MEM_MGR_HPP_
#define _PVFMM_MEM_MGR_HPP_

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(push,target(mic))
#endif
namespace pvfmm{
namespace mem{

/**
 * \brief Identify each type uniquely.
 */
template <class T>
class TypeTraits{

  public:

    static inline uintptr_t ID();

    static inline bool IsPOD();
};

/**
 * \brief MemoryManager class declaration.
 */
class MemoryManager{

  public:

    static const char init_mem_val=42;

    /**
     * \brief Header data for each memory block.
     */
    struct MemHead{
      size_t n_indx;
      size_t n_elem;
      uintptr_t type_id;
      uintptr_t type_size;
      unsigned char check_sum;
    };

    /**
     * \brief Constructor for MemoryManager.
     */
    MemoryManager(size_t N);

    /**
     * \brief Constructor for MemoryManager.
     */
    ~MemoryManager();

    static inline MemHead* GetMemHead(void* p);

    inline void* malloc(const size_t n_elem=1, const size_t type_size=sizeof(char)) const;

    inline void free(void* p) const;

    void print() const;

    static void test();

  private:

    // Private constructor
    MemoryManager();

    // Private copy constructor
    MemoryManager(const MemoryManager& m);

    // Check all free memory equals init_mem_val
    void Check() const;

    /**
     * \brief Node structure for a doubly linked list, representing free and
     * occupied memory blocks. Blocks are split, merged or state is changed
     * between free and occupied in O(1) time given the pointer to the MemNode.
     */
    struct MemNode{
      bool free;
      size_t size;
      char* mem_ptr;
      size_t prev, next;
      std::multimap<size_t, size_t>::iterator it;
    };

    /**
     * \brief Return index of one of the available MemNodes from node_stack or
     * create new MemNode by resizing node_buff.
     */
    inline size_t new_node() const;

    /**
     * \brief Add node index for now available MemNode to node_stack.
     */
    inline void delete_node(size_t indx) const;

    char* buff;          // pointer to memory buffer.
    size_t buff_size;    // total buffer size in bytes.
    size_t n_dummy_indx; // index of first (dummy) MemNode in link list.

    mutable std::vector<MemNode> node_buff;         // storage for MemNode objects, this can only grow.
    mutable std::stack<size_t> node_stack;          // stack of available free MemNodes from node_buff.
    mutable std::multimap<size_t, size_t> free_map; // pair (MemNode.size, MemNode_id) for all free MemNodes.
    mutable omp_lock_t omp_lock;                    // openmp lock to prevent concurrent changes.
};

/** A global MemoryManager object. This is the default for aligned_new and
 * aligned_free */
extern MemoryManager glbMemMgr;


inline uintptr_t align_ptr(uintptr_t ptr){
  static uintptr_t     ALIGN_MINUS_ONE=MEM_ALIGN-1;
  static uintptr_t NOT_ALIGN_MINUS_ONE=~ALIGN_MINUS_ONE;
  return ((ptr+ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE);
}

/**
 * \brief Aligned allocation as an alternative to new. Uses placement new to
 * construct objects.
 */
template <class T>
inline T* aligned_new(size_t n_elem=1, const MemoryManager* mem_mgr=&glbMemMgr);

/**
 * \brief Aligned de-allocation as an alternative to delete. Calls the object
 * destructors. Not sure which destructor is called for virtual classes, this
 * is why we also match the TypeTraits<T>::ID()
 */
template <class T>
inline void aligned_delete(T* A, const MemoryManager* mem_mgr=&glbMemMgr);

/**
 * \brief Wrapper to memcpy. Also checks if source and destination pointers are
 * the same.
 */
inline void * memcopy(void * destination, const void * source, size_t num);

}//end namespace
}//end namespace

#include <mem_mgr.txx>

#ifdef __INTEL_OFFLOAD
#pragma offload_attribute(pop)
#endif

#endif //_PVFMM_MEM_MGR_HPP_
