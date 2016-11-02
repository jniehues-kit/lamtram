#pragma once

#include <lamtram/sentence.h>
#include <lamtram/ll-stats.h>
#include <lamtram/builder-factory.h>
#include <lamtram/softmax-base.h>
#include <lamtram/dict-utils.h>
#include <lamtram/neural-lm.h>
#include <lamtram/multitask-model.h>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <vector>
#include <iostream>

namespace dynet {
class Model;
struct ComputationGraph;
struct LookupParameter;
struct RNNBuilder;
}

namespace lamtram {

class ExternCalculator;
typedef std::shared_ptr<ExternCalculator> ExternCalculatorPtr;

// A class for feed-forward neural network LMs
class SharedMultiTaskNeuralLM : public NeuralLM , public MultiTaskModel{
    friend class EncoderAttentional;

public:

    // Create a new SharedMultiTaskNeuralLM and add it to the existing model
    //   vocab: the vocabulary
    //   ngram_context: number of previous words to use (should be at least one)
    //   extern_context: The size in nodes of vector containing extern context
    //     that is calculated from something other than the previous words.
    //     Can be set to zero if this doesn't apply.
    //   extern_feed: Whether to feed the previous external context back in
    //   wordrep_size: The size of the word representations.
    //   unk_id: The ID of unknown words.
    //   softmax_sig: A signature indicating the type of softmax to use
    //   model: The model in which to store the parameters.
    SharedMultiTaskNeuralLM(const std::vector<DictPtr> & vocab, int ngram_context, int extern_context,
             bool extern_feed,
             int wordrep_size, const BuilderSpec & hidden_spec, int unk_id,
             const std::string & softmax_sig, bool word_embedding_in_softmax,
             int attention_context, bool source_word_embedding_in_softmax, int source_word_embedding_in_softmax_context,
             dynet::Model & model);

    SharedMultiTaskNeuralLM(const std::vector<DictPtr> & vocabs, int ngram_context, int extern_context,
             bool extern_feed,
             int wordrep_size, const BuilderSpec & hidden_spec, int unk_id,
             const std::string & softmax_sig,bool word_embedding_in_softmax,
             int attention_context, bool source_word_embedding_in_softmax, int source_word_embedding_in_softmax_context,
             ExternCalculatorPtr & att,
             dynet::Model & model);

    ~SharedMultiTaskNeuralLM() { }


    // Reading/writing functions
    static SharedMultiTaskNeuralLM* Read(const std::vector<DictPtr> & vocab, std::istream & in, dynet::Model & model);
    void Write(std::ostream & out);

    // Information functions
    static std::string ModelID() { return "sharedmultitasknlm"; }

    // Accessors
    int GetVocabSize() const { std::cerr << "Not supported since changing" << std::endl; exit(-1); return 0; };
    SoftmaxBase & GetSoftmax() { std::cerr << "Not supported since changing" << std::endl; exit(-1); return *softmax_; }

    // Setters
    void SetVocabulary(int sourceIndex, int targetIndex) {current_voc_ = sourceIndex; p_wr_W_ = ps_wr_W_[current_voc_];vocab_ = vocabs_[current_voc_];softmax_ = softmaxes_[current_voc_];vocab_ = vocabs_[current_voc_];};


protected:

    // The vocabulary
    std::vector<DictPtr> vocabs_;


    // Pointers to the parameters
    std::vector<dynet::LookupParameter> ps_wr_W_;

    // Pointer to the softmax
    std::vector<SoftmaxPtr> softmaxes_;
    
    int current_voc_;

};

typedef std::shared_ptr<SharedMultiTaskNeuralLM> SharedMultiTaskNeuralLMPtr;

}
