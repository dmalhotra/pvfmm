-include $(PVFMM_DIR)/MakeVariables

ifndef CXXFLAGS_PVFMM
-include ../MakeVariables
endif

ifndef CXXFLAGS_PVFMM
$(error Cannot find file: MakeVariables)
endif

RM = rm -f
MKDIRS = mkdir -p

BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
TMPDIR = ./tmp
INCDIR = ./include

TARGET_BIN = \
       $(BINDIR)/example1 \
       $(BINDIR)/example2 \
       $(BINDIR)/fmm_cheb

all : $(TARGET_BIN)

ifeq ($(INTEL_OFFLOAD_OK),yes)

$(BINDIR)/%: $(OBJDIR)/%.o
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) -no-offload      $^       $(LDFLAGS_PVFMM) -o $@
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  $^_mic   $(LDFLAGS_PVFMM) -o $@_mic
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) -D__MIC_ASYNCH__ $^_async $(LDFLAGS_PVFMM) -o $@_async

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) -no-offload      -I$(INCDIR) -c $^ -o $@
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  -I$(INCDIR) -c $^ -o $@_mic
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM) -D__MIC_ASYNCH__ -I$(INCDIR) -c $^ -o $@_async

else

$(BINDIR)/%: $(OBJDIR)/%.o
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  $^       $(LDFLAGS_PVFMM) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX_PVFMM) $(CXXFLAGS_PVFMM)                  -I$(INCDIR) -c $^ -o $@

endif

clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*
	$(RM) *~ */*~

