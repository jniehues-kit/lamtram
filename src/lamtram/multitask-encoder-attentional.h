#pragma once

#include <lamtram/sentence.h>
#include <lamtram/ll-stats.h>
#include <lamtram/linear-encoder.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/multitask-model.h>
#include <lamtram/neural-lm.h>
#include <lamtram/extern-calculator.h>
#include <lamtram/mapping.h>
#include <dynet/dynet.h>
#include <vector>
#include <iostream>
#include <lamtram/gru-cond.h>

namespace dynet {
class Model;
struct ComputationGraph;
struct Parameter;
struct RNNBuilder;
}

namespace lamtram {


// A class to calculate extern_calcal context
class MultiTaskExternAttentional : public ExternAttentional {
public:

    MultiTaskExternAttentional(const std::vector<LinearEncoderPtr> & encoders,
                      const std::string & attention_type, const std::string & attention_hist,
                      int state_size, const std::string & lex_type,
                      const std::vector<DictPtr> & vocab_src, const std::vector<DictPtr> & vocab_trg,
                     int attention_context, bool source_word_embedding_in_softmax, int source_word_embedding_in_softmax_context, std::string encoder_type,
                      dynet::Model & mod);
    virtual ~MultiTaskExternAttentional() { }


    // Reading/writing functions
    static MultiTaskExternAttentional* Read(std::istream & in, const std::vector<DictPtr> & vocab_src, const std::vector<DictPtr> & vocab_trg, std::vector<MultiTaskModelPtr> & mtmodels, dynet::Model & model);
    virtual void Write(std::ostream & out);



protected:

    const std::string encoder_type_;

};

typedef std::shared_ptr<MultiTaskExternAttentional> MultiTaskExternAttentionalPtr;



// A class for feed-forward neural network LMs
class MultiTaskEncoderAttentional : public EncoderAttentional {

public:

    // Create a new MultiTaskEncoderAttentional and add it to the existing model
    MultiTaskEncoderAttentional(const ExternAttentionalPtr & extern_calc,
                       const NeuralLMPtr & decoder, std::string decoder_type_,
                       dynet::Model & model);
    ~MultiTaskEncoderAttentional() { }

    // Reading/writing functions
    static MultiTaskEncoderAttentional* Read(const std::vector<DictPtr> & vocab_src, const std::vector<DictPtr> & vocab_trg, std::istream & in, std::vector<MultiTaskModelPtr> & mtmodels, dynet::Model & model);
    virtual void Write(std::ostream & out);

    static std::string ModelID() { return "multitaskencatt"; }


protected:


    const std::string decoder_type_;

};

typedef std::shared_ptr<MultiTaskEncoderAttentional> MultiTaskEncoderAttentionalPtr;

}
