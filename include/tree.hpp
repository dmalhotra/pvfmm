/**
 * \file tree.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-11-2010
 * \brief This file contains the definition of the base class for a tree.
 */

#include <cassert>
#include <vector>

#include <pvfmm_common.hpp>
#include <tree_node.hpp>


#ifndef _PVFMM_TREE_HPP_
#define _PVFMM_TREE_HPP_

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
  Tree(): dim(0), root_node(sctl::NullIterator<Node_t>()), max_depth(PVFMM_MAX_DEPTH) { };

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
  Node_t* RootNode() {return (root_node==sctl::NullIterator<Node_t>()?NULL:&root_node[0]);}

  /**
   * \brief Returns a new node of the same type as the root node.
   */
  sctl::Iterator<Node_t> NewNode() {assert(root_node!=sctl::NullIterator<Node_t>()); return sctl::Iterator<Node_t>(root_node->NewNode());}

  /**
   * \brief Returns a pointer to the first node in preorder traversal (the root
   * node).
   */
  sctl::Iterator<Node_t> PreorderFirst();

  /**
   * \brief Returns a pointer to the next node in preorder traversal.
   */
  sctl::Iterator<Node_t> PreorderNxt(sctl::Iterator<Node_t> curr_node);

  /**
   * \brief Returns a pointer to the first node in postorder traversal.
   */
  sctl::Iterator<Node_t> PostorderFirst();

  /**
   * \brief Returns a pointer to the next node in postorder traversal.
   */
  sctl::Iterator<Node_t> PostorderNxt(sctl::Iterator<Node_t> curr_node);

  /**
   * \brief Returns a list of all nodes in preorder traversal.
   */
  std::vector<sctl::Iterator<TreeNode>>& GetNodeList();

  /**
   * \brief Dimension of the tree.
   */
  int Dim() {return dim;}

 protected:

  int dim;              // dimension of the tree
  sctl::Iterator<Node_t> root_node;    // owning iterator for root node
  int max_depth;        // maximum tree depth
  std::vector<sctl::Iterator<TreeNode>> node_lst;
};

}//end namespace

#include <tree.txx>

#endif //_PVFMM_TREE_HPP_
