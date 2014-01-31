/**
 * \file tree.cpp
 * \author Dhairya Malhotra, dhairya.malhotra88@gmail.com
 * \date 12-11-2010
 * \brief This file contains the implementation of the class Tree.
 */

#include <tree.hpp>
#include <assert.h>

template <class TreeNode>
Tree<TreeNode>::~Tree(){
  if(RootNode()!=NULL){
    delete root_node;
  }
}

template <class TreeNode>
void Tree<TreeNode>::Initialize(typename Node_t::NodeData* init_data_){
  dim=init_data_->dim;
  max_depth=init_data_->max_depth;
  if(max_depth>MAX_DEPTH) max_depth=MAX_DEPTH;

  root_node=new Node_t();
  root_node->Initialize(NULL,0,init_data_);
}

template <class TreeNode>
void Tree<TreeNode>::RefineTree(){
  Node_t* curr_node;

  curr_node=PostorderFirst();
  while(curr_node!=NULL){
    if(!curr_node->IsLeaf())
      if(!curr_node->SubdivCond()) curr_node->Truncate();
    curr_node=PostorderNxt(curr_node);
  }

  curr_node=PreorderFirst();
  while(curr_node!=NULL){
    if(curr_node->IsLeaf())
      if(curr_node->SubdivCond())
        curr_node->Subdivide();
    curr_node=PreorderNxt(curr_node);
  }
}

template <class TreeNode>
TreeNode* Tree<TreeNode>::PreorderFirst(){
  return root_node;
}

template <class TreeNode>
TreeNode* Tree<TreeNode>::PreorderNxt(Node_t* curr_node){
  assert(curr_node!=NULL);

  int n=(1UL<<dim);
  if(!curr_node->IsLeaf())
    for(int i=0;i<n;i++)
      if(curr_node->Child(i)!=NULL)
        return (Node_t*)curr_node->Child(i);

  Node_t* node=curr_node;
  while(true){
    int i=node->Path2Node()+1;
    node=(Node_t*)node->Parent();
    if(node==NULL) return NULL;

    for(;i<n;i++)
      if(node->Child(i)!=NULL)
        return (Node_t*)node->Child(i);
  }
}

template <class TreeNode>
TreeNode* Tree<TreeNode>::PostorderFirst(){
  Node_t* node=root_node;

  int n=(1UL<<dim);
  while(true){
    if(node->IsLeaf()) return node;
    for(int i=0;i<n;i++)
      if(node->Child(i)!=NULL){
        node=(Node_t*)node->Child(i);
        break;
      }
  }
}

template <class TreeNode>
TreeNode* Tree<TreeNode>::PostorderNxt(Node_t* curr_node){
  assert(curr_node!=NULL);
  Node_t* node=curr_node;

  int j=node->Path2Node()+1;
  node=(Node_t*)node->Parent();
  if(node==NULL) return NULL;

  int n=(1UL<<dim);
  for(;j<n;j++){
    if(node->Child(j)!=NULL){
      node=(Node_t*)node->Child(j);
      while(true){
        if(node->IsLeaf()) return node;
        for(int i=0;i<n;i++)
          if(node->Child(i)!=NULL){
            node=(Node_t*)node->Child(i);
            break;
          }
      }
    }
  }

  return node;
}

template <class TreeNode>
std::vector<TreeNode*>& Tree<TreeNode>::GetNodeList(){
  if(root_node->GetStatus() & 1){
    node_lst.clear();
    TreeNode* n=this->PreorderFirst();
    while(n!=NULL){
      int& status=n->GetStatus();
      status=(status & (~(int)1));
      node_lst.push_back(n);
      n=this->PreorderNxt(n);
    }
  }
  return node_lst;
}

