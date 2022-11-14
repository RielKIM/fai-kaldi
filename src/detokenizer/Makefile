
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

OBJFILES = detokenizer.o

LIBNAME = kaldi-detokenizer

ADDLIBS = ../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
