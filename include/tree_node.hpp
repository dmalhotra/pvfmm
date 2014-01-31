/**
 * \file tree_node.hpp
 * \author Dhairya Malhotra, dhairya.malhotra88@gmail.com
 * \date 12-10-2010
 * \brief This file contains the definition of a virtual base class for a tree
 * node.
 */

#ifndef _TREE_NODE_HPP_
#define _TREE_NODE_HPP_

#include <pvfmm_common.hpp>
#include <assert.h>
#include <cstring>

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
  TreeNode(): dim(0), depth(0), max_depth(MAX_DEPTH), parent(NULL), child(NULL), status(1) { }

  /**
   * \brief Virtual destructor
   */
  virtual ~TreeNode();

  /**
   * \brief Initialize the node by passing the relevant data.
   */
  virtual void Initialize(TreeNode* parent_, int path2node_, NodeData* data_) ;

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
  bool IsLeaf(){return child == NULL;}

  /**
   * \brief Returns the child corresponding to the input parameter.
   */
  TreeNode* Child(int id);

  /**
   * \brief Returns a pointer to the parent node.
   */
  TreeNode* Parent();

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
  virtual TreeNode* NewNode(TreeNode* n_=NULL);

  /**
   * \brief Evaluates and returns the subdivision condition for this node.
   * 'true' if node requires further subdivision.
   */
  virtual bool SubdivCond();

  /**
   * \brief Create child nodes and Initialize them.
   */
  virtual void Subdivide() ;

  /**
   * \brief Truncates the tree i.e. makes this a leaf node.
   */
  virtual void Truncate() ;

  /**
   * \brief Set the parent for this node.
   */
  void SetParent(TreeNode* p, int path2node_) ;

  /**
   * \brief Set a child for this node.
   */
  void SetChild(TreeNode* c, int id) ;

  /**
   * \brief Returns status.
   */
  int& GetStatus();

  /**
   * \brief Update status for all nodes up to the root node.
   */
  void SetStatus(int flag);

 protected:

  int dim;               //Dimension of the tree
  int depth;             //Depth of the node (root -> 0)
  int max_depth;         //Maximum depth
  int path2node;         //Identity among siblings
  TreeNode* parent;      //Pointer to parent node
  TreeNode** child;      //Pointer child nodes
  int status;

};

#endif

