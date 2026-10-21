ACLOCAL_AMFLAGS = -I m4 $(ACLOCAL_FLAGS)
AUTOMAKE_OPTIONS = subdir-objects foreign
@SET_MAKE@

include MakeVariables.in
CXXFLAGS = $(CXXFLAGS_PVFMM)
LDLIBS = $(LDLIBS_PVFMM)

# Suppress autoconf's @DEFS@ (PACKAGE_*, HAVE_*_H, HAVE_LIB*, LT_OBJDIR, ...).
# With config.h removed these would otherwise be dumped on every compile line,
# and none of them are referenced by pvfmm or SCTL.
DEFS =

INSTALL_DIR = $(DESTDIR)$(datadir)/$(PACKAGE)
EX_DIR = ./examples
RESULT_DIR = ./result

EXTRA_DIST = \
									version.txt \
									$(EX_DIR)/src \
									$(EX_DIR)/include \
									$(EX_DIR)/Makefile \
									Doxyfile \
									scripts \
									SCTL/include

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
									include/pvfmm_common.hpp \
									include/pvfmm.hpp \
									include/pvfmm.txx \
									include/cheb_node.hpp \
									include/cheb_utils.hpp \
									include/device_wrapper.hpp \
									include/fmm_cheb.hpp \
									include/fmm_node.hpp \
									include/fmm_pts.hpp \
									include/fmm_pts_gpu.hpp \
									include/fmm_tree.hpp \
									include/interac_list.hpp \
									include/kernel.hpp \
									include/mat_utils.hpp \
									include/mpi_node.hpp \
									include/mpi_tree.hpp \
									include/precomp_mat.hpp \
									include/tree.hpp \
									include/tree_node.hpp \
									include/cheb_node.txx \
									include/cheb_utils.txx \
									include/device_wrapper.txx \
									include/fmm_cheb.txx \
									include/fmm_node.txx \
									include/fmm_pts.txx \
									include/fmm_tree.txx \
									include/interac_list.txx \
									include/kernel.txx \
									include/mpi_node.txx \
									include/mpi_tree.txx \
									include/precomp_mat.txx \
									include/tree.txx \
									include/tree_node.txx

lib_libpvfmm_la_HEADERS = \
									$(pvfmm_cpp_header_files) \
									include/pvfmm.f90 \
									include/pvfmm.h

# the sources to add to the library and to add to the source distribution
lib_libpvfmm_la_SOURCES = \
									$(pvfmm_cpp_header_files) \
									src/pvfmm-wrapper.cpp

if NVCC_OK
lib_libpvfmm_la_SOURCES += \
													src/fmm_pts_gpu.cu
endif

# Bake the numerical deps into libpvfmm.so so it is self-contained for dlopen
# consumers (python/ ctypes, julia/ Libdl); executables also get these via
# LDLIBS_PVFMM, but a dlopen'd .so must resolve its own symbols.
lib_libpvfmm_la_LIBADD = @FFTW_LIB@ @FFTWF_LIB@ @FFTWL_LIB@ @LAPACK_LIBS@ @BLAS_LIBS@ @FLIBS@ @CUDA_LDLIBS@ @LIBMVEC_LIB@ @OPENMP_LIB@

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

if HAVE_DOXYGEN
doxygen-doc:
	$(DOXYGEN) $(srcdir)/Doxyfile
	@if test -d docs/latex; then \
	  echo "Building PDF (docs/latex/refman.pdf)..."; \
	  $(MAKE) -C docs/latex || echo "Warning: PDF build failed (need pdflatex/texlive); HTML docs are in docs/html." >&2; \
	fi
else
doxygen-doc:
	@echo "Error: doxygen not found; install doxygen to build the documentation." >&2; exit 1
endif


###############################################################################
# INSTALL/UNINSTALL SCTL headers: the whole SCTL/include tree, next to the
# PVFMM headers (no file list here to go stale when SCTL changes)
#------------------------------------------------------------------------------

install-sctl-headers:
	for f in `cd $(srcdir)/SCTL/include && find . -type f ! -name '.*'`; do \
	  $(MKDIR_P) "$(DESTDIR)$(pkgincludedir)/`dirname $$f`" && \
	  $(INSTALL_DATA) "$(srcdir)/SCTL/include/$$f" "$(DESTDIR)$(pkgincludedir)/$$f" || exit 1; \
	done

uninstall-sctl-headers:
	$(RM) -r "$(DESTDIR)$(pkgincludedir)/sctl" "$(DESTDIR)$(pkgincludedir)/sctl.hpp"


###############################################################################
# INSTALL/UNINSTALL MakeVariables
#------------------------------------------------------------------------------

# The installed SCTL headers are in the PVFMM include dir, so drop the -I for
# the SCTL submodule.
install-make-variables:
	$(MKDIR_P) $(INSTALL_DIR)
	sed -e '/(TOP_SRCDIR_PVFMM)\/SCTL\/include/d' \
		-e 's/(TOP_SRCDIR_PVFMM)/(INSTALL_DIR_PVFMM)/' \
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

install-data-local: install-sctl-headers install-make-variables install-libtool

uninstall-local: uninstall-sctl-headers uninstall-make-variables uninstall-libtool


###############################################################################
# LOCAL CLEAN
#------------------------------------------------------------------------------

clean-local:
	cd $(EX_DIR) && $(MAKE) clean;
	$(RM) -r $(RESULT_DIR)/*
	$(RM) *~ */*~ */*/*~
	$(RM) ./lib/*

#------------------------------------------------------------------------------

