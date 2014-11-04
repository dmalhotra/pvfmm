ACLOCAL_AMFLAGS = -I m4 $(ACLOCAL_FLAGS)
AUTOMAKE_OPTIONS = subdir-objects
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

DIST_SUBDIRS= \
									doc/doc

EXTRA_DIST = \
									$(EX_DIR)/src \
									$(EX_DIR)/include \
									$(EX_DIR)/Makefile \
									$(DX_CONFIG) \
									scripts

# additional include pathes necessary to compile the C++ library
AM_CXXFLAGS =
AM_LDFLAGS = -static

###############################################################################
# THE LIBRARIES TO BUILD
###############################################################################

# the library names to build (note we are building static libs only)
pkglib_LIBRARIES = lib/libpvfmm.a

# where to install the headers on the system
lib_libfmm_adir = $(pkgincludedir)

# the list of header files that belong to the library (to be installed later)
lib_libfmm_a_HEADERS = \
									include/pvfmm_config.h \
									include/pvfmm_common.hpp \
									include/pvfmm.hpp \
									include/blas.h \
									include/cheb_node.hpp \
									include/cheb_utils.hpp \
									include/device_wrapper.hpp \
									include/dtypes.h \
									include/fft_wrapper.hpp \
									include/fmm_cheb.hpp \
									include/fmm_gll.hpp \
									include/fmm_node.hpp \
									include/fmm_pts.hpp \
									include/fmm_pts_gpu.hpp \
									include/fmm_tree.hpp \
									include/interac_list.hpp \
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
									include/quad_utils.hpp \
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
									include/mat_utils.txx \
									include/mem_mgr.txx \
									include/mortonid.txx \
									include/mpi_node.txx \
									include/mpi_tree.txx \
									include/ompUtils.txx \
									include/parUtils.txx \
									include/precomp_mat.txx \
									include/quad_utils.txx \
									include/tree.txx \
									include/vector.txx

#nodist_lib_libpvfmm_a_HEADERS = \
#									include/pvfmm_config.h

# the sources to add to the library and to add to the source distribution
lib_libpvfmm_a_SOURCES = \
									$(lib_libpvfmm_a_HEADERS) \
									src/device_wrapper.cpp \
									src/fmm_gll.cpp \
									src/legendre_rule.cpp \
									src/mem_mgr.cpp \
									src/mortonid.cpp \
									src/profile.cpp \
									src/tree_node.cpp

if NVCC_OK
lib_libpvfmm_a_SOURCES += \
													src/fmm_pts_gpu.cu
endif

dist_noinst_SCRIPTS = autogen.sh

core: $(pkglib_LIBRARIES)

all: #all-docs all-examples 

###############################################################################
# COMPILE .cu FILES
#------------------------------------------------------------------------------

if NVCC_OK

.cu.o :
	$(NVCC_PVFMM) $(NVCCFLAGS_PVFMM) -c $< -o $@

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

pdf-doc:
	cd doc/doc && $(MAKE);

all-docs: doxygen-doc pdf-doc

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

install-data-local: install-make-variables install-doxygen-doc
	cd doc/doc && $(MAKE) install;

uninstall-local: uninstall-make-variables uninstall-doxygen-doc
	cd doc/doc && $(MAKE) uninstall;


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

