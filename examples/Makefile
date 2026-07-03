ifndef PVFMM_DIR
PVFMM_DIR=./..
endif

-include $(PVFMM_DIR)/MakeVariables

ifndef CXX_PVFMM
$(error Cannot find file: MakeVariables)
endif

CXX=$(CXX_PVFMM)
ifneq ($(CC_PVFMM),)
CC=$(CC_PVFMM)
endif
ifneq ($(FC_PVFMM),)
FC=$(FC_PVFMM)
endif
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
       $(BINDIR)/example-sctl \
       $(BINDIR)/kernel-test \
       $(BINDIR)/fmm_pts \
       $(BINDIR)/fmm_cheb

ifneq ($(CC_PVFMM),)
TARGET_BIN += \
       $(BINDIR)/example-c \
       $(BINDIR)/example2-c
endif
ifneq ($(FC_PVFMM),)
TARGET_BIN += \
       $(BINDIR)/example-f \
       $(BINDIR)/example2-f
endif

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

