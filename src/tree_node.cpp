/**
 * \file tree_node.cpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-10-2010
 * \brief This file contains the implementation of the class TreeNode.
 */

#include <tree_node.hpp>
#include <assert.h>
#include <iostream>
#include <mem_mgr.hpp>

namespace pvfmm{

TreeNode::~TreeNode(){
  if(!child) return;
  int n=(1UL<<dim);
  //Delete the children.
  for(int i=0;i<n;i++){
    if(child[i]!=NULL)
      mem::aligned_delete(child[i]);
  }
  mem::aligned_delete(child);
  child=NULL;
}

void TreeNode::Initialize(TreeNode* parent_, int path2node_, NodeData* data_){
  parent=parent_;
  depth=(parent==NULL?0:parent->Depth()+1);
  if(data_!=NULL){
    dim=data_->dim;
    max_depth=data_->max_depth;
    if(max_depth>MAX_DEPTH) max_depth=MAX_DEPTH;
  }else if(parent!=NULL){
    dim=parent->dim;
    max_depth=parent->max_depth;
  }

  assert(path2node_>=0 && path2node_<(1U<<dim));
  path2node=path2node_;

  //assert(parent_==NULL?true:parent_->Child(path2node_)==this);
}

TreeNode* TreeNode::Child(int id){
  assert(id<(1<<dim));
  if(child==NULL) return NULL;
  return child[id];
}

TreeNode* TreeNode::Parent(){
  return parent;
}

int TreeNode::Path2Node(){
  return path2node;
}

TreeNode* TreeNode::NewNode(TreeNode* n_){
  TreeNode* n=(n_==NULL?mem::aligned_new<TreeNode>():n_);
  n->dim=dim;
  n->max_depth=max_depth;
  return n_;
}

bool TreeNode::SubdivCond(){
  if(!IsLeaf()){
    int n=(1UL<<dim);
    for(int i=0;i<n;i++){
      TreeNode* ch=this->Child(i);
      assert(ch!=NULL); //This should never happen
      if(!ch->IsLeaf()) return true;
    }
    if(Depth()>=max_depth) return false;
    return true;
  }else{
    if(this->Depth()>=max_depth) return false;
    return false;
  }
}

void TreeNode::Subdivide() {
  if(child) return;
  SetStatus(1);
  int n=(1UL<<dim);
  child=mem::aligned_new<TreeNode*>(n);
  for(int i=0;i<n;i++){
    child[i]=this->NewNode();
    child[i]->parent=this;
    child[i]->Initialize(this,i,NULL);
  }
}

void TreeNode::Truncate() {
  if(!child) return;
  SetStatus(1);
  int n=(1UL<<dim);
  for(int i=0;i<n;i++){
    if(child[i]!=NULL)
      delete child[i];
  }
  delete[] child;
  child=NULL;
}

void TreeNode::SetParent(TreeNode* p, int path2node_) {
  assert(path2node_>=0 && path2node_<(1<<dim));
  assert(p==NULL?true:p->Child(path2node_)==this);

  parent=p;
  path2node=path2node_;
  depth=(parent==NULL?0:parent->Depth()+1);
  if(parent!=NULL) max_depth=parent->max_depth;
}

void TreeNode::SetChild(TreeNode* c, int id) {
  assert(id<(1<<dim));
  //assert(child!=NULL);
  //if(child[id]!=NULL)
  //  delete child[id];
  child[id]=c;
  if(c!=NULL) child[id]->SetParent(this,id);
}

int& TreeNode::GetStatus(){
  return status;
}

void TreeNode::SetStatus(int flag){
  status=(status|flag);
  if(parent && !(parent->GetStatus() & flag))
    parent->SetStatus(flag);
}

}//end namespace
