/**
 * \file tree_node.cpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-10-2010
 * \brief This file contains the implementation of the class TreeNode.
 */

#include <mpi.h>
#include <tree_node.hpp>
#include <assert.h>
#include <iostream>


namespace pvfmm{

TreeNode::~TreeNode(){
  if(child==sctl::NullIterator<sctl::Iterator<TreeNode>>()) return;
  int n=(1UL<<dim);
  //Delete the children.
  for(int i=0;i<n;i++){
    if(child[i]!=sctl::NullIterator<TreeNode>())
      sctl::aligned_delete(child[i]);
  }
  sctl::aligned_delete(child);
  child=sctl::NullIterator<sctl::Iterator<TreeNode>>();
}

void TreeNode::Initialize(sctl::Iterator<TreeNode> parent_, int path2node_, NodeData* data_){
  parent=parent_;
  depth=(parent==sctl::NullIterator<TreeNode>()?0:parent->Depth()+1);
  if(data_!=NULL){
    dim=data_->dim;
    max_depth=data_->max_depth;
    if(max_depth>PVFMM_MAX_DEPTH) max_depth=PVFMM_MAX_DEPTH;
  }else if(parent!=sctl::NullIterator<TreeNode>()){
    dim=parent->dim;
    max_depth=parent->max_depth;
  }

  assert(path2node_>=0 && path2node_<(int)(1U<<dim));
  path2node=path2node_;

  //assert(parent_==NULL?true:parent_->Child(path2node_)==this);
}

sctl::Iterator<TreeNode> TreeNode::Child(int id){
  assert(id<(1<<dim));
  if(child==sctl::NullIterator<sctl::Iterator<TreeNode>>()) return sctl::NullIterator<TreeNode>();
  return child[id];
}

sctl::Iterator<TreeNode> TreeNode::Parent(){
  return parent;
}

int TreeNode::Path2Node(){
  return path2node;
}

sctl::Iterator<TreeNode> TreeNode::NewNode(sctl::Iterator<TreeNode> n_){
  sctl::Iterator<TreeNode> n=(n_==sctl::NullIterator<TreeNode>()?sctl::aligned_new<TreeNode>():n_);
  n->dim=dim;
  n->max_depth=max_depth;
  return n;
}

bool TreeNode::SubdivCond(){
  if(!IsLeaf()){
    int n=(1UL<<dim);
    for(int i=0;i<n;i++){
      sctl::Iterator<TreeNode> ch=this->Child(i);
      assert(ch!=sctl::NullIterator<TreeNode>()); //This should never happen
      if(!ch->IsLeaf()) return true;
    }
    if(Depth()>=max_depth) return false;
    return true;
  }else{
    if(this->Depth()>=max_depth) return false;
    return false;
  }
}

void TreeNode::Subdivide(sctl::Iterator<TreeNode> self_) {
  if(child!=sctl::NullIterator<sctl::Iterator<TreeNode>>()) return;
  assert(&self_[0]==this); // self_ must be this node's own allocation iterator
  SetStatus(1);
  int n=(1UL<<dim);
  child=sctl::aligned_new<sctl::Iterator<TreeNode>>(n);
  for(int i=0;i<n;i++){
    child[i]=this->NewNode();
    child[i]->Initialize(self_,i,NULL);
  }
}

void TreeNode::Truncate() {
  if(child==sctl::NullIterator<sctl::Iterator<TreeNode>>()) return;
  SetStatus(1);
  int n=(1UL<<dim);
  for(int i=0;i<n;i++){
    if(child[i]!=sctl::NullIterator<TreeNode>())
      sctl::aligned_delete(child[i]);
  }
  sctl::aligned_delete(child);
  child=sctl::NullIterator<sctl::Iterator<TreeNode>>();
}

void TreeNode::SetParent(sctl::Iterator<TreeNode> p, int path2node_) {
  assert(path2node_>=0 && path2node_<(1<<dim));
  assert(p==sctl::NullIterator<TreeNode>()?true:&p->Child(path2node_)[0]==this);

  parent=p;
  path2node=path2node_;
  depth=(parent==sctl::NullIterator<TreeNode>()?0:parent->Depth()+1);
  if(parent!=sctl::NullIterator<TreeNode>()) max_depth=parent->max_depth;
}

void TreeNode::SetChild(sctl::Iterator<TreeNode> c, int id) {
  assert(id<(1<<dim));
  //assert(child!=sctl::NullIterator<sctl::Iterator<TreeNode>>());
  //if(child[id]!=sctl::NullIterator<TreeNode>())
  //  sctl::aligned_delete(child[id]);
  child[id]=c;
  assert(parent!=sctl::NullIterator<TreeNode>());
  if(c!=sctl::NullIterator<TreeNode>()) child[id]->SetParent(parent->child[path2node],id);
}

int& TreeNode::GetStatus(){
  return status;
}

void TreeNode::SetStatus(int flag){
  status=(status|flag);
  if(parent!=sctl::NullIterator<TreeNode>() && !(parent->GetStatus() & flag))
    parent->SetStatus(flag);
}

}//end namespace
