-include $(PVFMM_DIR)/MakeVariables

ifndef CXX_PVFMM
-include ../MakeVariables
endif

ifndef CXXFLAGS_PVFMM
$(error Cannot find file: MakeVariables)
endif

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

ifeq ($(INTEL_OFFLOAD_OK),yes)

$(BINDIR)/%: $(OBJDIR)/%.o
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) -no-offload         $^       $(LDLIBS) -o $@
	$(CXX) $(CXXFLAGS)                     $^_async $(LDLIBS) -o $@_async
	$(CXX) $(CXxFLAGS) -D__DEVICE_SYNC__=1 $^_mic   $(LDLIBS) -o $@_mic

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) -no-offload         -I$(INCDIR) -c $^ -o $@
	$(CXX) $(CXXFLAGS)                     -I$(INCDIR) -c $^ -o $@_async
	$(CXX) $(CXXFLAGS) -D__DEVICE_SYNC__=1 -I$(INCDIR) -c $^ -o $@_mic

else

$(BINDIR)/%: $(OBJDIR)/%.o
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $^ -o $@

endif

clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*
	$(RM) *~ */*~

