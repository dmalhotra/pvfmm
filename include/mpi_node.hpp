/**
 * \file mpi_node.hpp
 * \author Dhairya Malhotra, dhairya.malhotra88@gmail.com
 * \date 12-10-2010
 * \brief This file contains the definition of a virtual base class for a
 * locally essential tree node.
 */

#ifndef _MPI_NODE_HPP_
#define _MPI_NODE_HPP_

#include <pvfmm_common.hpp>
#include <assert.h>
#include <tree_node.hpp>
#include <mortonid.hpp>
#include <vector.hpp>

/**
 * \brief A structure for storing packed data for transmitting a node to
 * another MPI process.
 */
struct PackedData{
  size_t length;//Length of data
  void* data;   //Pointer to data
};

/**
 * \brief Virtual base class for a locally essential tree node.
 */
template <class T>
class MPI_Node: public TreeNode{

 public:

  typedef T Real_t;

  /**
   * \brief Base class for node data. Contains initialization data for the node.
   */
  class NodeData: public TreeNode::NodeData{

   public:

     size_t max_pts;
     Vector<Real_t> pt_coord;
     Vector<Real_t> pt_value;
  };

  /**
   * \brief Initialize.
   */
  MPI_Node(): TreeNode(){ghost=false;}

  /**
   * \brief Virtual destructor.
   */
  virtual ~MPI_Node();

  /**
   * \brief Initialize the node with relevant data.
   */
  virtual void Initialize(TreeNode* parent_, int path2node_, TreeNode::NodeData*) ;

  /**
   * \brief Returns list of coordinate and value vectors which need to be
   * sorted and partitioned across MPI processes and the scatter index is
   * saved.
   */
  virtual void NodeDataVec(std::vector<Vector<Real_t>*>& coord,
                           std::vector<Vector<Real_t>*>& value,
                           std::vector<Vector<size_t>*>& scatter){
    coord  .push_back(&pt_coord  );
    value  .push_back(&pt_value  );
    scatter.push_back(&pt_scatter);
  }

  /**
   * \brief Clear node data.
   */
  virtual void ClearData();

  /**
   * \brief Returns the colleague corresponding to the input index.
   */
  MPI_Node<Real_t>* Colleague(int index){return colleague[index];}

  /**
   * \brief Set the colleague corresponding to the input index.
   */
  void SetColleague(MPI_Node<Real_t>* node_, int index){colleague[index]=node_;}

  /**
   * \brief Returns the cost of this node. Used for load balancing.
   */
  virtual Real_t NodeCost(){return 1.0;}

  /**
   * \brief Returns an array of size dim containing the coordinates of the
   * node.
   */
  Real_t* Coord(){assert(coord!=NULL); return coord;}

  /**
   * \brief Determines if the node is a Ghost node or not.
   */
  bool IsGhost(){return ghost;}

  /**
   * \brief Sets the ghost flag of this node.
   */
  void SetGhost(bool x){ghost=x;}

  /**
   * \brief Gets Morton Id of this node.
   */
  inline MortonId GetMortonId();

  /**
   * \brief Sets the coordinates of this node using the given Morton Id.
   */
  inline void SetCoord(MortonId& mid);

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
  virtual void Subdivide();

  /**
   * \brief Truncates the tree i.e. makes this a leaf node.
   */
  virtual void Truncate();

  /**
   * \brief Pack this node to be transmitted to another process. The node
   * is responsible for allocating and freeing the memory for the actual data.
   */
  virtual PackedData Pack(bool ghost=false, void* buff_ptr=NULL, size_t offset=0);

  /**
   * \brief Initialize the node with data from another process.
   */
  virtual void Unpack(PackedData data, bool own_data=true);

  /**
   * \brief Read source distribution at points on a grid defined by array of x,
   * y and z coordinates.
   */
  virtual void ReadVal(std::vector<Real_t> x,std::vector<Real_t> y, std::vector<Real_t> z, Real_t* val, bool show_ghost=true);

  /**
   * \brief Append node VTU data to vectors.
   */
  virtual void VTU_Data(std::vector<Real_t>& coord, std::vector<Real_t>& value, std::vector<int32_t>& connect, std::vector<int32_t>& offset, std::vector<uint8_t>& types, int lod=-1);

  Vector<Real_t> pt_coord;   //coordinates of points
  Vector<Real_t> pt_value;   //value at points
  Vector<size_t> pt_scatter; //scatter index mapping original data.

 protected:

  bool ghost;
  size_t max_pts;

  Real_t coord[COORD_DIM];
  MPI_Node<Real_t>* colleague[COLLEAGUE_COUNT];
  Vector<char> packed_data;
};

#include <mpi_node.txx>

#endif

