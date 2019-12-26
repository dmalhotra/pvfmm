ifndef PVFMM_DIR
PVFMM_DIR=./..
endif

-include $(PVFMM_DIR)/MakeVariables

ifndef CXX_PVFMM
$(error Cannot find file: MakeVariables)
endif

# FC=$(FC_PVFMM) # TODO: for now, FC must be provided by user
# CC=$(CC_PVFMM) # TODO: for now, CC must be provided by user
CXX=$(CXX_PVFMM)
CXXFLAGS=$(CXXFLAGS_PVFMM)
LDLIBS=$(LDLIBS_PVFMM)

RM = rm -f
MKDIRS = mkdir -p

BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./include

TARGET_BIN = \
       $(BINDIR)/example1 \
       $(BINDIR)/example2 \
       $(BINDIR)/fmm_pts \
       $(BINDIR)/fmm_cheb

all : $(TARGET_BIN)

$(BINDIR)/%: $(SRCDIR)/%.f90
	-@$(MKDIRS) $(dir $@)
	$(PVFMM_DIR)/libtool --mode=link --tag=FC $(FC) $(CXXFLAGS) -I$(INCDIR) $^ $(LDLIBS) -o $@

$(BINDIR)/%: $(SRCDIR)/%.c
	-@$(MKDIRS) $(dir $@)
	$(PVFMM_DIR)/libtool --mode=link --tag=CC $(CC) $(CXXFLAGS) -I$(INCDIR) $^ $(LDLIBS) -o $@

$(BINDIR)/%: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(PVFMM_DIR)/libtool --mode=link --tag=CXX $(CXX) $(CXXFLAGS) -I$(INCDIR) $^ $(LDLIBS) -o $@

clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*
	$(RM) *~ */*~

