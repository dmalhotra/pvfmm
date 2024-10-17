ACLOCAL_AMFLAGS = -I m4 $(ACLOCAL_FLAGS)
AUTOMAKE_OPTIONS = subdir-objects foreign
@SET_MAKE@

include MakeVariables.in
CXXFLAGS = $(CXXFLAGS_PVFMM)
LDLIBS = $(LDLIBS_PVFMM)

# DOXYGEN SUPPORT
include aminclude.am

INSTALL_DIR = $(DESTDIR)$(datadir)/$(PACKAGE)
EX_DIR = ./examples
RESULT_DIR = ./result

# the subdirectories of the project to go into
#SUBDIRS = 

#DIST_SUBDIRS = doc/doc


EXTRA_DIST = \
									$(EX_DIR)/src \
									$(EX_DIR)/include \
									$(EX_DIR)/Makefile \
									$(DX_CONFIG) \
									scripts

# additional include paths necessary to compile the C++ library
AM_CXXFLAGS =
#AM_LDFLAGS = -static

###############################################################################
# THE LIBRARIES TO BUILD
###############################################################################

# the library names to build
pkglib_LTLIBRARIES = lib/libpvfmm.la

# where to install the headers on the system
lib_libpvfmm_ladir = $(pkgincludedir)

# the list of header files that belong to the library (to be installed later)
pvfmm_cpp_header_files = \
									include/pvfmm_config.h \
									include/pvfmm_common.hpp \
									include/pvfmm.hpp \
									include/pvfmm.txx \
									include/blas.h \
									include/cheb_node.hpp \
									include/cheb_utils.hpp \
									include/device_wrapper.hpp \
									include/dtypes.h \
									include/fft_wrapper.hpp \
									include/fmm_cheb.hpp \
									include/fmm_node.hpp \
									include/fmm_pts.hpp \
									include/fmm_pts_gpu.hpp \
									include/fmm_tree.hpp \
									include/interac_list.hpp \
									include/intrin_wrapper.hpp \
									include/kernel.hpp \
									include/lapack.h \
									include/legendre_rule.hpp \
									include/matrix.hpp \
									include/mat_utils.hpp \
									include/mem_mgr.hpp \
									include/mortonid.hpp \
									include/mpi_node.hpp \
									include/mpi_tree.hpp \
									include/ompUtils.h \
									include/parUtils.h \
									include/precomp_mat.hpp \
									include/profile.hpp \
									include/stacktrace.h \
									include/tree.hpp \
									include/tree_node.hpp \
									include/vector.hpp \
									include/cheb_node.txx \
									include/cheb_utils.txx \
									include/device_wrapper.txx \
									include/fmm_cheb.txx \
									include/fmm_node.txx \
									include/fmm_pts.txx \
									include/fmm_tree.txx \
									include/interac_list.txx \
									include/kernel.txx \
									include/matrix.txx \
									include/mem_mgr.txx \
									include/mortonid.txx \
									include/mpi_node.txx \
									include/mpi_tree.txx \
									include/ompUtils.txx \
									include/parUtils.txx \
									include/precomp_mat.txx \
									include/tree.txx \
									include/vector.txx \
									SCTL/include/sctl.hpp

lib_libpvfmm_la_HEADERS = \
									$(pvfmm_cpp_header_files) \
									include/pvfmm.f90 \
									include/pvfmm.h

subdirheadersdir = $(includedir)/pvfmm/sctl

subdirheaders_HEADERS = \
												SCTL/include/sctl/blas.h \
												SCTL/include/sctl/boundary_integral.hpp \
												SCTL/include/sctl/boundary_integral.txx \
												SCTL/include/sctl/boundary_quadrature.hpp \
												SCTL/include/sctl/cheb_utils.hpp \
												SCTL/include/sctl/comm.hpp \
												SCTL/include/sctl/comm.txx \
												SCTL/include/sctl/common.hpp \
												SCTL/include/sctl/complex.hpp \
												SCTL/include/sctl/complex.txx \
												SCTL/include/sctl/fft_wrapper.hpp \
												SCTL/include/sctl/fft_wrapper.txx \
												SCTL/include/sctl/fmm-wrapper.hpp \
												SCTL/include/sctl/fmm-wrapper.txx \
												SCTL/include/sctl/generic-kernel.hpp \
												SCTL/include/sctl/generic-kernel.txx \
												SCTL/include/sctl/intrin-wrapper.hpp \
												SCTL/include/sctl/iterator.hpp \
												SCTL/include/sctl/iterator.txx \
												SCTL/include/sctl/kernel_functions.hpp \
												SCTL/include/sctl/lagrange-interp.hpp \
												SCTL/include/sctl/lagrange-interp.txx \
												SCTL/include/sctl/lapack.h \
												SCTL/include/sctl/lin-solve.hpp \
												SCTL/include/sctl/lin-solve.txx \
												SCTL/include/sctl/mat_utils.hpp \
												SCTL/include/sctl/mat_utils.txx \
												SCTL/include/sctl/math_utils.hpp \
												SCTL/include/sctl/math_utils.txx \
												SCTL/include/sctl/matrix.hpp \
												SCTL/include/sctl/matrix.txx \
												SCTL/include/sctl/mem_mgr.hpp \
												SCTL/include/sctl/mem_mgr.txx \
												SCTL/include/sctl/morton.hpp \
												SCTL/include/sctl/morton.txx \
												SCTL/include/sctl/ode-solver.hpp \
												SCTL/include/sctl/ode-solver.txx \
												SCTL/include/sctl/ompUtils.hpp \
												SCTL/include/sctl/ompUtils.txx \
												SCTL/include/sctl/permutation.hpp \
												SCTL/include/sctl/permutation.txx \
												SCTL/include/sctl/profile.hpp \
												SCTL/include/sctl/profile.txx \
												SCTL/include/sctl/quadrule.hpp \
												SCTL/include/sctl/quadrule.txx \
												SCTL/include/sctl/sph_harm.hpp \
												SCTL/include/sctl/sph_harm.txx \
												SCTL/include/sctl/sse2neon.h \
												SCTL/include/sctl/stacktrace.h \
												SCTL/include/sctl/static-array.hpp \
												SCTL/include/sctl/static-array.txx \
												SCTL/include/sctl/tensor.hpp \
												SCTL/include/sctl/tensor.txx \
												SCTL/include/sctl/tree.hpp \
												SCTL/include/sctl/tree.txx \
												SCTL/include/sctl/vec-test.hpp \
												SCTL/include/sctl/vec.hpp \
												SCTL/include/sctl/vec.txx \
												SCTL/include/sctl/vector.hpp \
												SCTL/include/sctl/vector.txx \
												SCTL/include/sctl/vtudata.hpp \
												SCTL/include/sctl/vtudata.txx

#nodist_lib_libpvfmm_la_HEADERS = \
#									include/pvfmm_config.h

# the sources to add to the library and to add to the source distribution
lib_libpvfmm_la_SOURCES = \
									$(pvfmm_cpp_header_files) \
									src/cheb_utils.cpp \
									src/device_wrapper.cpp \
									src/legendre_rule.cpp \
									src/mem_mgr.cpp \
									src/mortonid.cpp \
									src/profile.cpp \
									src/pvfmm-wrapper.cpp \
									src/tree_node.cpp

if NVCC_OK
lib_libpvfmm_la_SOURCES += \
													src/fmm_pts_gpu.cu
endif

dist_noinst_SCRIPTS = autogen.sh

core: $(pkglib_LTLIBRARIES)

all: #all-docs all-examples 

###############################################################################
# COMPILE .cu FILES
#------------------------------------------------------------------------------

if NVCC_OK

.cu.o :
	$(NVCC_PVFMM) $(NVCCFLAGS_PVFMM) -c $< -o $@

.cu.lo :
	$(LIBTOOL) --tag=CC --mode=compile $(NVCC_PVFMM) $(NVCCFLAGS_PVFMM) -c $< -o $@

endif


###############################################################################
# BUILD EXAMPLES
#------------------------------------------------------------------------------

all-examples: core
	cd $(EX_DIR) && $(MAKE);

$(EX_DIR)/bin/%: core
	cd $(EX_DIR) && $(MAKE) bin/$(notdir $@)


###############################################################################
# BUILD DOCUMENTATION
#------------------------------------------------------------------------------

all-docs: doxygen-doc

###############################################################################
# INSTALL/UNINSTALL MakeVariables, pvfmm_config.h
#------------------------------------------------------------------------------

install-make-variables:
	$(MKDIR_P) $(INSTALL_DIR)
	sed -e 's/(TOP_SRCDIR_PVFMM)/(INSTALL_DIR_PVFMM)/' \
		-e "s/PKG_SUBDIR_PVFMM =/PKG_SUBDIR_PVFMM = \/$(PACKAGE)/" \
		MakeVariables > $(INSTALL_DIR)/MakeVariables

uninstall-make-variables:
	$(RM) $(INSTALL_DIR)/MakeVariables


###############################################################################
# INSTALL/UNINSTALL libtool
#------------------------------------------------------------------------------

install-libtool:
	$(MKDIR_P) $(INSTALL_DIR)
	$(INSTALL_PROGRAM) libtool $(INSTALL_DIR)/libtool

uninstall-libtool:
	$(RM) $(INSTALL_DIR)/libtool


###############################################################################
# INSTALL/UNINSTALL DOCS
#------------------------------------------------------------------------------

if DOXYGEN_OK

install-doxygen-doc: doxygen-doc
	for doxygen_dir in $(srcdir)/doc/doxygen/*/ ; do \
		doxygen_dir=$${doxygen_dir%*/}; \
		dest_dir=$(INSTALL_DIR)/doc/$${doxygen_dir##*/}; \
		$(MKDIR_P) $${dest_dir}; \
		$(INSTALL_DATA) $${doxygen_dir}/* $${dest_dir}; done;

uninstall-doxygen-doc:
	$(RM) -r $(INSTALL_DIR)/doc/*
	$(RM) -r $(INSTALL_DIR)/doc

else

install-doxygen-doc:

uninstall-doxygen-doc:

endif


###############################################################################
# LOCAL INSTALL/UNINSTALL
#------------------------------------------------------------------------------

install: install-am
	@echo
	@echo '======================================================================='
	@echo 'You should add the following to your ~/.bashrc'
	@echo '    export PVFMM_DIR=$(INSTALL_DIR)'
	@echo
	@echo 'You will need to load $$(PVFMM_DIR)/MakeVariables in your Makefile and'
	@echo 'use the variables CXXFLAGS_PVFMM and LDLIBS_PVFMM for compiling your code.'
	@echo '======================================================================='

install-data-local: install-make-variables install-libtool install-doxygen-doc

uninstall-local: uninstall-make-variables uninstall-libtool uninstall-doxygen-doc


###############################################################################
# LOCAL CLEAN
#------------------------------------------------------------------------------

clean-doxygen:
	$(RM) -r $(DX_CLEANFILES)

clean-local: clean-doxygen
	cd $(EX_DIR) && $(MAKE) clean;
	$(RM) -r $(RESULT_DIR)/*
	$(RM) *~ */*~ */*/*~
	$(RM) ./lib/*

#------------------------------------------------------------------------------

