compilePdfOptions= #-interaction=nonstopmode
compile= ${compPdftex}
compPdftex=pdflatex ${compilePdfOptions}

TARGET = doc.pdf
TEXINFO_TEX = doc.tex
EXTRA_DIST = doc.tex bibsync.bib
INSTALL_DIR = $(DESTDIR)$(datadir)/$(PACKAGE)

###############################################################################
# COMPILE DOCS
#------------------------------------------------------------------------------

if PDFLATEX_OK
filename=doc
doc.pdf: ${SOURCES}
	${compile} ${filename}
	-bibtex ${filename}
	-bibtex ${filename}
	${compile} ${filename}
	${compile} ${filename}
	${compile} ${filename}
	@echo Done ....!


###############################################################################
# INSTALL DOCS
#------------------------------------------------------------------------------

install-pdf-doc: doc.pdf
	$(MKDIR_P) $(INSTALL_DIR)/doc;
	$(INSTALL_DATA) ${TARGET} $(INSTALL_DIR)/doc/${TARGET};

uninstall-pdf-doc:
	$(RM) $(INSTALL_DIR)/doc/${TARGET}

else

install-pdf-doc:

uninstall-pdf-doc:

endif


###############################################################################
# LOCAL INSTALL/UNINSTALL
#------------------------------------------------------------------------------

install-data-local: install-pdf-doc

uninstall-local: uninstall-pdf-doc


###############################################################################
# LOCAL CLEAN
#------------------------------------------------------------------------------

clean-local:
	rm -r -f *.aux *.log *.dvi *.blg *.bbl *.pdf tmp/*
	@echo Cleaned ....! 

#------------------------------------------------------------------------------
