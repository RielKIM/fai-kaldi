
all:

include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

#BINFILES = online-decoder

BASEOBJS = $(wildcard ../base/*.o)
MATRIXOBJS = ../matrix/compressed-matrix.o ../matrix/kaldi-matrix.o ../matrix/kaldi-vector.o ../matrix/matrix-functions.o \
            ../matrix/numpy-array.o ../matrix/optimization.o ../matrix/packed-matrix.o ../matrix/sp-matrix.o \
            ../matrix/srfft.o ../matrix/tp-matrix.o
UTILSOBJS = $(wildcard ../util/*.o)
TREEOBJS = $(wildcard ../tree/*.o)
GMMOBJS = $(wildcard ../gmm/*.o)
TRANSOBJS = $(wildcard ../transform/*.o)
FEATBJS = $(wildcard ../feat/*.o)
HMMOBJS = $(wildcard ../hmm/*.o)
FSTOBJS = $(wildcard ../fstext/*.o)
LATOBJS = $(wildcard ../lat/*.o)
DECODEROBJS = $(wildcard ../decoder/*.o)
CUDAMATRIXOBJS = $(wildcard ../cudamatrix/*.o)
NNET2OBJS = $(wildcard ../nnet2/*.o)
CHAINOBJS = $(wildcard ../chain/*.o)
NNET3OBJS = $(wildcard ../nnet3/*.o)
IVECTOROBJS = $(wildcard ../ivector/*.o)
ONLINE2OBJS = $(wildcard ../online2/*.o)
DETOKENOBJS = $(wildcard ../detokenizer/*.o)
ONLINEDECODEROBJS = $(wildcard ../onlinedecoder/*.o)

OBJFILES = online-decoder.o
#OBJFILES += $(BASEOBJS) \
            $(MATRIXOBJS) \
            $(UTILSOBJS) \
            $(TREEOBJS) \
            $(GMMOBJS) \
            $(TRANSOBJS) \
            $(FEATBJS) \
            $(HMMOBJS) \
            $(FSTOBJS) \
            $(LATOBJS) \
            $(DECODEROBJS) \
            $(CUDAMATRIXOBJS) \
            $(NNET2OBJS) \
            $(CHAINOBJS) \
            $(NNET3OBJS) \
            $(IVECTOROBJS) \
            $(ONLINE2OBJS) \
            $(DETOKENOBJS) \
            $(ONLINEDECODEROBJS)

LIBNAME = online-decoder

TESTFILES =

ADDLIBS = ../onlinedecoder/kaldi-online-decoder.a ../detokenizer/kaldi-detokenizer.a ../online2/kaldi-online2.a ../ivector/kaldi-ivector.a \
          ../nnet3/kaldi-nnet3.a ../chain/kaldi-chain.a ../nnet2/kaldi-nnet2.a \
          ../cudamatrix/kaldi-cudamatrix.a ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a ../hmm/kaldi-hmm.a \
          ../feat/kaldi-feat.a ../transform/kaldi-transform.a \
          ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a ../util/kaldi-util.a \
          ../matrix/kaldi-matrix.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
