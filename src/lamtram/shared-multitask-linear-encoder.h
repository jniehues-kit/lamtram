#pragma once

#include <lamtram/sentence.h>
#include <lamtram/linear-encoder.h>
#include <lamtram/ll-stats.h>
#include <lamtram/builder-factory.h>
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

// A class for feed-forward neural network LMs
class SharedMultiTaskLinearEncoder : public LinearEncoder , public MultiTaskModel{
    friend class EncoderAttentional;

public:

    // Create a new SharedMultiTaskLinearEncoder and add it to the existing model
    SharedMultiTaskLinearEncoder(const std::vector<int> & vocab_size, int wordrep_size,
             const BuilderSpec & hidden_spec, int unk_id,
             dynet::Model & model);
    ~SharedMultiTaskLinearEncoder() { }


    // Reading/writing functions
    static SharedMultiTaskLinearEncoder* Read(std::istream & in, dynet::Model & model);
    virtual void Write(std::ostream & out);

    static std::string ModelID() { return "sharedmultitasklinearencoder"; }

    // Accessors
    int GetVocabSize() const { std::cerr << "Get VocabSize not supported for multi-task" << std::endl; exit(-1); }
    
    void SetVocabulary(int sourceIndex, int targetIndex) {current_voc_ = sourceIndex; p_wr_W_ = ps_wr_W_[current_voc_];};

protected:

    std::vector<int> vocab_sizes_;

    // Pointers to the parameters
    std::vector<dynet::LookupParameter> ps_wr_W_; // Wordrep weights

    int current_voc_;


};

typedef std::shared_ptr<SharedMultiTaskLinearEncoder> SharedMultiTaskLinearEncoderPtr;

}
