/**
 * \file tree_node.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-10-2010
 * \brief This file contains the definition of a virtual base class for a tree
 * node.
 */

#include <pvfmm_common.hpp>
#include <sctl.hpp>

#ifndef _PVFMM_TREE_NODE_HPP_
#define _PVFMM_TREE_NODE_HPP_

namespace pvfmm{

/**
 * \brief Virtual base class for tree node.
 */
class TreeNode{

 public:

  /**
   * \brief Base class for node data. Contains initialization data for the node.
   */
  class NodeData{

   public:
     virtual ~NodeData(){};

     virtual void Clear(){}

     int max_depth;
     int dim;
  };

  /**
   * \brief Initialize pointers to NULL
   */
  TreeNode(): dim(0), depth(0), max_depth(PVFMM_MAX_DEPTH), parent(sctl::NullIterator<TreeNode>()), child(sctl::NullIterator<sctl::Iterator<TreeNode>>()), status(1) { }

  /**
   * \brief Virtual destructor
   */
  virtual ~TreeNode();

  /**
   * \brief Initialize the node by passing the relevant data.
   */
  virtual void Initialize(sctl::Iterator<TreeNode> parent_, int path2node_, NodeData* data_) ;

  /**
   * \brief Clear node data.
   */
  virtual void ClearData(){}

  /**
   * \brief Returns the dimension of the tree.
   */
  int Dim(){return dim;}

  /**
   * \brief Returns the depth of this node. (Root has depth 0)
   */
  int Depth(){return depth;}

  /**
   * \brief Returns 'true' if this is a leaf node.
   */
  bool IsLeaf(){return child == sctl::NullIterator<sctl::Iterator<TreeNode>>();}

  /**
   * \brief Returns the child corresponding to the input parameter.
   */
  /**
   * \brief Returns the stored (owning, full-length) allocation iterator of
   * the child node, as carried forward from its allocation in Subdivide.
   */
  sctl::Iterator<TreeNode> Child(int id);

  /**
   * \brief Returns the iterator for the parent node.
   */
  sctl::Iterator<TreeNode> Parent();

  /**
   * \brief Returns the index which corresponds to this node among its
   * siblings (parent's children).
   * this->Parent()->Child(this->Path2Node())==this
   */
  int Path2Node();

  /**
   * \brief Allocate a new object of the same type (as the derived class) and
   * return a pointer to it type cast as (TreeNode*).
   */
  virtual sctl::Iterator<TreeNode> NewNode(sctl::Iterator<TreeNode> n_=sctl::NullIterator<TreeNode>());

  /**
   * \brief Evaluates and returns the subdivision condition for this node.
   * 'true' if node requires further subdivision.
   */
  virtual bool SubdivCond();

  /**
   * \brief Create child nodes and Initialize them. `self_` is this node's
   * own allocation iterator (C++ erases it at the member-function boundary,
   * so the caller must supply it). It is copied into the children's `parent`
   * so node iterators are carried forward from allocation, never
   * reconstructed from raw pointers.
   */
  virtual void Subdivide(sctl::Iterator<TreeNode> self_) ;

  /**
   * \brief Truncates the tree i.e. makes this a leaf node.
   */
  virtual void Truncate() ;

  /**
   * \brief Set the parent for this node.
   */
  void SetParent(sctl::Iterator<TreeNode> p, int path2node_) ;

  /**
   * \brief Set a child for this node.
   */
  void SetChild(sctl::Iterator<TreeNode> c, int id) ;

  /**
   * \brief Returns status.
   */
  int& GetStatus();

  /**
   * \brief Update status for all nodes up to the root node.
   */
  void SetStatus(int flag);

  size_t node_id; //For translating node pointer to index.

 protected:

  int dim;               //Dimension of the tree
  int depth;             //Depth of the node (root -> 0)
  int max_depth;         //Maximum depth
  int path2node;         //Identity among siblings
  sctl::Iterator<TreeNode> parent;      //Iterator for parent node (non-owning backreference)
  sctl::Iterator<sctl::Iterator<TreeNode>> child;   //Owning iterators for child nodes (array)
  int status;

};

}//end namespace

#endif //_PVFMM_TREE_NODE_HPP_
