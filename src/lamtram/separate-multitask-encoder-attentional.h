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
class SeparateMultiTaskExternAttentional : public ExternAttentional, public MultiTaskModel {
public:

    SeparateMultiTaskExternAttentional(const std::vector<LinearEncoderPtr> & encoders,
                      const std::string & attention_type, const std::string & attention_hist,
                      int state_size, const std::string & lex_type,
                      const std::vector<DictPtr> & vocab_src, const std::vector<DictPtr> & vocab_trg,
                     int attention_context, bool source_word_embedding_in_softmax, int source_word_embedding_in_softmax_context, std::string encoder_type,
                      dynet::Model & mod);
    virtual ~SeparateMultiTaskExternAttentional() { }


    // Reading/writing functions
    static SeparateMultiTaskExternAttentional* Read(std::istream & in, const std::vector<DictPtr> & vocab_src, const std::vector<DictPtr> & vocab_trg, std::vector<MultiTaskModelPtr> & mtmodels, dynet::Model & model);
    virtual void Write(std::ostream & out);

    // Setters
    void SetVocabulary(int sourceIndex, int targetIndex) {current_source_voc_ = sourceIndex; current_target_voc_ = targetIndex;
                if(attention_type_ == "bilin") {
                    p_e_ehid_W_ = p_e_ehid_Ws_[current_source_voc_][current_target_voc_];
                } else if(attention_type_.substr(0,4) == "mlp:") {
                    p_ehid_h_W_ = p_ehid_h_Ws_[current_source_voc_][current_target_voc_];
                    p_ehid_state_W_ = p_ehid_state_Ws_[current_source_voc_][current_target_voc_];
                    p_e_ehid_W_ = p_e_ehid_Ws_[current_source_voc_][current_target_voc_];
                }else if(attention_type_.substr(0,6) == "mlp_b:") {
                    p_ehid_h_W_ = p_ehid_h_Ws_[current_source_voc_][current_target_voc_];
                    p_ehid_state_W_ = p_ehid_state_Ws_[current_source_voc_][current_target_voc_];
                    p_ehid_h_b_ = p_ehid_h_bs_[current_source_voc_][current_target_voc_];
                    p_e_ehid_W_ = p_e_ehid_Ws_[current_source_voc_][current_target_voc_];
                    p_e_ehid_b_ = p_e_ehid_bs_[current_source_voc_][current_target_voc_];
                    
                }
    }


protected:

    const std::string encoder_type_;

    int current_source_voc_;
    int current_target_voc_;

    // Parameters
    std::vector<std::vector<dynet::Parameter> > p_ehid_h_Ws_;
    std::vector<std::vector<dynet::Parameter> > p_ehid_h_bs_;
    std::vector<std::vector<dynet::Parameter> > p_ehid_state_Ws_;
    std::vector<std::vector<dynet::Parameter> > p_e_ehid_Ws_;
    std::vector<std::vector<dynet::Parameter> > p_e_ehid_bs_;
    std::vector<std::vector<dynet::Parameter> > p_align_sum_Ws_;

};

typedef std::shared_ptr<SeparateMultiTaskExternAttentional> SeparateMultiTaskExternAttentionalPtr;



// A class for feed-forward neural network LMs
class SeparateMultiTaskEncoderAttentional : public EncoderAttentional, public MultiTaskModel {

public:

    // Create a new SeparateMultiTaskEncoderAttentional and add it to the existing model
    SeparateMultiTaskEncoderAttentional(const ExternAttentionalPtr & extern_calc,
                       const NeuralLMPtr & decoder, std::string decoder_type_,
                       int source_voc_size,int target_voc_size,
                       dynet::Model & model);
    ~SeparateMultiTaskEncoderAttentional() { }

    // Reading/writing functions
    static SeparateMultiTaskEncoderAttentional* Read(const std::vector<DictPtr> & vocab_src, const std::vector<DictPtr> & vocab_trg, std::istream & in, std::vector<MultiTaskModelPtr> & mtmodels, dynet::Model & model);
    virtual void Write(std::ostream & out);

    static std::string ModelID() { return "separatemultitaskencatt"; }


    // Setters
    void SetVocabulary(int sourceIndex, int targetIndex) {current_source_voc_ = sourceIndex; current_target_voc_ = targetIndex;
        p_enc2dec_W_ = p_enc2dec_Ws_[current_source_voc_][current_target_voc_];
        p_enc2dec_b_ = p_enc2dec_bs_[current_source_voc_][current_target_voc_];
    }

protected:


    const std::string decoder_type_;

    int current_source_voc_;
    int current_target_voc_;


    // Parameters
    std::vector<std::vector<dynet::Parameter> > p_enc2dec_Ws_; // Encoder to decoder weights
    std::vector<std::vector<dynet::Parameter> > p_enc2dec_bs_; // Encoder to decoder bias


};

typedef std::shared_ptr<SeparateMultiTaskEncoderAttentional> SeparateMultiTaskEncoderAttentionalPtr;

}
