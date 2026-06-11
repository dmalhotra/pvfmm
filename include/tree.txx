/**
 * \file tree.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-11-2010
 * \brief This file contains the implementation of the class Tree.
 */

namespace pvfmm{

template <class TreeNode>
Tree<TreeNode>::~Tree(){
  if(root_node!=sctl::NullIterator<Node_t>()){
    sctl::aligned_delete(root_node);
  }
}

template <class TreeNode>
void Tree<TreeNode>::Initialize(typename Node_t::NodeData* init_data_){
  dim=init_data_->dim;
  max_depth=init_data_->max_depth;
  if(max_depth>PVFMM_MAX_DEPTH) max_depth=PVFMM_MAX_DEPTH;

  if(root_node!=sctl::NullIterator<Node_t>()) sctl::aligned_delete(root_node);
  root_node=sctl::aligned_new<Node_t>();
  root_node->Initialize(sctl::NullIterator<::pvfmm::TreeNode>(),0,init_data_);
}

template <class TreeNode>
void Tree<TreeNode>::RefineTree(){
  sctl::Iterator<Node_t> curr_node;

  curr_node=PostorderFirst();
  while(curr_node!=sctl::NullIterator<Node_t>()){
    if(!curr_node->IsLeaf())
      if(!curr_node->SubdivCond()) curr_node->Truncate();
    curr_node=PostorderNxt(curr_node);
  }

  curr_node=PreorderFirst();
  while(curr_node!=sctl::NullIterator<Node_t>()){
    if(curr_node->IsLeaf())
      if(curr_node->SubdivCond())
        curr_node->Subdivide((sctl::Iterator<::pvfmm::TreeNode>)curr_node);
    curr_node=PreorderNxt(curr_node);
  }
}

template <class TreeNode>
sctl::Iterator<TreeNode> Tree<TreeNode>::PreorderFirst(){
  return root_node;
}

template <class TreeNode>
sctl::Iterator<TreeNode> Tree<TreeNode>::PreorderNxt(sctl::Iterator<Node_t> curr_node){
  assert(curr_node!=sctl::NullIterator<Node_t>());

  int n=(1UL<<dim);
  if(!curr_node->IsLeaf())
    for(int i=0;i<n;i++)
      if(curr_node->Child(i)!=sctl::NullIterator<::pvfmm::TreeNode>())
        return (sctl::Iterator<Node_t>)curr_node->Child(i);

  sctl::Iterator<Node_t> node=curr_node;
  while(true){
    int i=node->Path2Node()+1;
    auto par=node->Parent();
    if(par==sctl::NullIterator<::pvfmm::TreeNode>()) return sctl::NullIterator<Node_t>();
    node=(sctl::Iterator<Node_t>)par;

    for(;i<n;i++)
      if(node->Child(i)!=sctl::NullIterator<::pvfmm::TreeNode>())
        return (sctl::Iterator<Node_t>)node->Child(i);
  }
}

template <class TreeNode>
sctl::Iterator<TreeNode> Tree<TreeNode>::PostorderFirst(){
  sctl::Iterator<Node_t> node=root_node;

  int n=(1UL<<dim);
  while(true){
    if(node->IsLeaf()) return node;
    for(int i=0;i<n;i++)
      if(node->Child(i)!=sctl::NullIterator<::pvfmm::TreeNode>()){
        node=(sctl::Iterator<Node_t>)node->Child(i);
        break;
      }
  }
}

template <class TreeNode>
sctl::Iterator<TreeNode> Tree<TreeNode>::PostorderNxt(sctl::Iterator<Node_t> curr_node){
  assert(curr_node!=sctl::NullIterator<Node_t>());
  sctl::Iterator<Node_t> node=curr_node;

  int j=node->Path2Node()+1;
  auto par=node->Parent();
  if(par==sctl::NullIterator<::pvfmm::TreeNode>()) return sctl::NullIterator<Node_t>();
  node=(sctl::Iterator<Node_t>)par;

  int n=(1UL<<dim);
  for(;j<n;j++){
    if(node->Child(j)!=sctl::NullIterator<::pvfmm::TreeNode>()){
      node=(sctl::Iterator<Node_t>)node->Child(j);
      while(true){
        if(node->IsLeaf()) return node;
        for(int i=0;i<n;i++)
          if(node->Child(i)!=sctl::NullIterator<::pvfmm::TreeNode>()){
            node=(sctl::Iterator<Node_t>)node->Child(i);
            break;
          }
      }
    }
  }

  return node;
}

template <class TreeNode>
std::vector<sctl::Iterator<TreeNode>>& Tree<TreeNode>::GetNodeList(){
  if(root_node->GetStatus() & 1){
    node_lst.clear();
    sctl::Iterator<Node_t> n=this->PreorderFirst();
    while(n!=sctl::NullIterator<Node_t>()){
      int& status=n->GetStatus();
      status=(status & (~(int)1));
      node_lst.push_back(n);
      n=this->PreorderNxt(n);
    }
  }
  return node_lst;
}

}//end namespace
