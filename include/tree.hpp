/**
 * \file tree.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-11-2010
 * \brief This file contains the definition of the base class for a tree.
 */

#ifndef _PVFMM_TREE_HPP_
#define _PVFMM_TREE_HPP_

#include <pvfmm_common.hpp>
#include <iostream>
#include <mem_mgr.hpp>

namespace pvfmm{

/**
 * \brief Base class for tree.
 */
template <class TreeNode>
class Tree{

 public:

   typedef TreeNode Node_t;

  /**
   * \brief Constructor.
   */
  Tree(): dim(0), root_node(NULL), max_depth(MAX_DEPTH), memgr(DEVICE_BUFFER_SIZE*1024l*1024l) { };

  /**
   * \brief Virtual destructor.
   */
  virtual ~Tree();

  /**
   * \brief Initialize the tree using initialization data for the root.
   */
  virtual void Initialize(typename Node_t::NodeData* init_data) ;

  /**
   * \brief Subdivide or truncate nodes based on SubdivCond().
   */
  virtual void RefineTree();

  /**
   * \brief Returns a pointer to the root node.
   */
  Node_t* RootNode() {return root_node;}

  /**
   * \brief Returns a new node of the same type as the root node.
   */
  Node_t* NewNode() {assert(root_node!=NULL); return (Node_t*)root_node->NewNode();}

  /**
   * \brief Returns a pointer to the first node in preorder traversal (the root
   * node).
   */
  Node_t* PreorderFirst();

  /**
   * \brief Returns a pointer to the next node in preorder traversal.
   */
  Node_t* PreorderNxt(Node_t* curr_node);

  /**
   * \brief Returns a pointer to the first node in postorder traversal.
   */
  Node_t* PostorderFirst();

  /**
   * \brief Returns a pointer to the next node in postorder traversal.
   */
  Node_t* PostorderNxt(Node_t* curr_node);

  /**
   * \brief Returns a list of all nodes in preorder traversal.
   */
  std::vector<TreeNode*>& GetNodeList();

  /**
   * \brief Dimension of the tree.
   */
  int Dim() {return dim;}

 protected:

  int dim;              // dimension of the tree
  Node_t* root_node;    // pointer to root node
  int max_depth;        // maximum tree depth
  std::vector<TreeNode*> node_lst;
  mem::MemoryManager memgr;
};

}//end namespace

#include <tree.txx>

#endif //_PVFMM_TREE_HPP_
