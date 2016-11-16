####### Library info

LIBCPP = \
    lamtram-train.cc \
    lamtram-train-multitask.cc \
    lamtram.cc \
    ensemble-decoder.cc \
    ensemble-classifier.cc \
    neural-lm.cc \
    linear-encoder.cc \
    encoder-decoder.cc \
    encoder-attentional.cc \
    encoder-classifier.cc \
    timer.cc \
    macros.cc \
    mapping.cc \
    classifier.cc \
    builder-factory.cc \
    model-utils.cc \
    counts.cc \
    input-file-stream.cc \
    softmax-full.cc \
    softmax-multilayer.cc \
    softmax-hinge.cc \
    softmax-mod.cc \
    softmax-diff.cc \
    softmax-class.cc \
    softmax-factory.cc \
    dict-utils.cc \
    dist-base.cc \
    dist-factory.cc \
    dist-ngram.cc \
    dist-one-hot.cc \
    dist-train.cc \
    dist-uniform.cc \
    dist-unk.cc \
    eval-measure-loader.cc \
    eval-measure-bleu.cc \
    eval-measure-extern.cc \
    eval-measure-ribes.cc \
    eval-measure-wer.cc \
    eval-measure-interp.cc \
    eval-measure-charf.cc \
    eval-measure-char.cc \
    eval-measure-fscore.cc \
    eval-measure.cc \
    gru-cond.cc \
    rnn-cond.cc \
    lstm-cond.cc \
    multitask-encoder-attentional.cc \
    shared-multitask-linear-encoder.cc \
    shared-multitask-neural-lm.cc \
    separate-multitask-linear-encoder.cc \
    separate-multitask-neural-lm.cc

lamtramincludedir = $(includedir)/lamtram
lamtraminclude_HEADERS = \
    lamtram-train.h \
    lamtram-train-multitask.h \
    lamtram.h \
    ensemble-decoder.h \
    ensemble-classifier.h \
    neural-lm.h \
    linear-encoder.h \
    encoder-decoder.h \
    encoder-attentional.h \
    encoder-classifier.h \
    timer.h \
    macros.h \
    mapping.h \
    classifier.h \
    builder-factory.h \
    model-utils.h \
    counts.h \
    input-file-stream.h \
    loss-stats.h \
    ll-stats.h \
    sentence.h \
    softmax-full.h \
    softmax-multilayer.h \
    softmax-hinge.h \
    softmax-mod.h \
    softmax-diff.h \
    softmax-class.h \
    softmax-factory.h \
    dict-utils.h \
    dist-base.h \
    dist-factory.h \
    dist-ngram.h \
    dist-one-hot.h \
    dist-train.h \
    dist-uniform.h \
    dist-unk.h \
    eval-measure-loader.h \
    eval-measure-bleu.h \
    eval-measure-extern.h \
    eval-measure-ribes.h \
    eval-measure-wer.h \
    eval-measure-interp.h \
    eval-measure-charf.h \
    eval-measure-char.h \
    eval-measure-fscore.h \
    eval-measure.h \
    gru-cond.h \
    rnn-cond.h \
    lstm-cond.h \
    multitask-encoder-attentional.h \
    shared-multitask-linear-encoder.h \
    shared-multitask-neural-lm.h \
    separate-multitask-linear-encoder.h \
    separate-multitask-neural-lm.h

AM_CXXFLAGS = $(BOOST_CPPFLAGS) $(EIGEN_CPPFLAGS) $(DYNET_CPPFLAGS) $(OPENMP_CXXFLAGS) -I$(srcdir)/..

lib_LTLIBRARIES = liblamtram.la

liblamtram_la_SOURCES = ${LIBCPP}
liblamtram_la_LDFLAGS = -version-info 0:0:0

####### Binary info

LDADD=liblamtram.la \
    $(DYNET_LDFLAGS) \
    $(BOOST_LDFLAGS) \
    $(BOOST_SYSTEM_LIB) \
    $(BOOST_PROGRAM_OPTIONS_LIB) \
    $(BOOST_SERIALIZATION_LIB) \
    $(BOOST_IOSTREAMS_LIB) \
    $(OPENMP_CXXFLAGS)

bin_PROGRAMS = lamtram-train-multitask lamtram-train lamtram dist-train

lamtram_train_SOURCES = lamtram-train-main.cc
lamtram_train_LDADD = $(LDADD)

lamtram_SOURCES = lamtram-main.cc
lamtram_LDADD = $(LDADD)

dist_train_SOURCES = dist-train-main.cc
dist_train_LDADD = $(LDADD)

lamtram_train_multitask_SOURCES = lamtram-train-multitask-main.cc
lamtram_train_multitask_LDADD = $(LDADD)
