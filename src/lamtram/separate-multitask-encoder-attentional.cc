#include <lamtram/separate-multitask-encoder-attentional.h>
#include <lamtram/shared-multitask-neural-lm.h>
#include <lamtram/shared-multitask-linear-encoder.h>
#include <lamtram/separate-multitask-neural-lm.h>
#include <lamtram/separate-multitask-linear-encoder.h>
#include <lamtram/macros.h>
#include <lamtram/builder-factory.h>
#include <dynet/model.h>
#include <dynet/nodes.h>
#include <dynet/rnn.h>
#include <dynet/dict.h>
#include <boost/range/irange.hpp>
#include <boost/algorithm/string.hpp>
#include <ctime>
#include <fstream>

using namespace std;
using namespace lamtram;


SeparateMultiTaskExternAttentional::SeparateMultiTaskExternAttentional(const std::vector<LinearEncoderPtr> & encoders,
                   const std::string & attention_type, const std::string & attention_hist, int state_size,
                   const std::string & lex_type,
                   const vector<DictPtr> & vocab_src, const vector<DictPtr> & vocab_trg,
                   int attention_context, bool source_word_embedding_in_softmax, int source_word_embedding_in_softmax_context, string encoder_type,
                   dynet::Model & mod): ExternAttentional(encoders,attention_type,attention_hist,state_size,lex_type,vocab_src[0],vocab_trg[0],attention_context,source_word_embedding_in_softmax,source_word_embedding_in_softmax_context,mod),encoder_type_(encoder_type) {
                     


  if(attention_type == "dot") {
    // No parameters for dot product
  } else if(attention_type_ == "bilin") {
    for(int i = 0; i < vocab_src.size(); i++) {
      p_e_ehid_Ws_.push_back(vector<dynet::Parameter>());
      for(int j = 0; j < vocab_trg.size(); j++) {
        if(i == 0 && j == 0) {
          p_e_ehid_Ws_[i].push_back(p_ehid_h_W_);
        }else{
          p_ehid_h_Ws_[i].push_back(mod.add_parameters({(unsigned int)state_size_, (unsigned int)context_size_}));
        }
      }
    }
  } else if(attention_type_.substr(0,4) == "mlp:") {
    hidden_size_ = stoi(attention_type_.substr(4));
    if(hidden_size_ == 0) hidden_size_ = GlobalVars::layer_size;
    use_bias_ = false;

    for(int i = 0; i < vocab_src.size(); i++) {
      p_ehid_h_Ws_.push_back(vector<dynet::Parameter>());
      p_ehid_state_Ws_.push_back(vector<dynet::Parameter>());
      p_e_ehid_Ws_.push_back(vector<dynet::Parameter>());
      for(int j = 0; j < vocab_trg.size(); j++) {
        cout << "Create Attention calcualtor" << i << " " << j << endl;
        if(i == 0 && j == 0) {
          p_ehid_h_Ws_[i].push_back(p_ehid_h_W_);
          p_ehid_state_Ws_[i].push_back(p_ehid_state_W_);
          p_e_ehid_Ws_[i].push_back(p_e_ehid_W_);
        }else{
          p_ehid_h_Ws_[i].push_back(mod.add_parameters({(unsigned int)hidden_size_, (unsigned int)context_size_}));
          p_ehid_state_Ws_[i].push_back(mod.add_parameters({(unsigned int)hidden_size_, (unsigned int)state_size_}));
          p_e_ehid_Ws_[i].push_back(mod.add_parameters({1, (unsigned int)hidden_size_}));
        }
      }
    }

  } else if(attention_type_.substr(0,6) == "mlp_b:") {
    hidden_size_ = stoi(attention_type_.substr(6));
    assert(hidden_size_ != 0);
    use_bias_ = true;

    for(int i = 0; i < vocab_src.size(); i++) {
      p_ehid_h_Ws_.push_back(vector<dynet::Parameter>());
      p_ehid_state_Ws_.push_back(vector<dynet::Parameter>());
      p_ehid_h_bs_.push_back(vector<dynet::Parameter>());
      p_e_ehid_Ws_.push_back(vector<dynet::Parameter>());
      p_e_ehid_bs_.push_back(vector<dynet::Parameter>());
      for(int j = 0; j < vocab_trg.size(); j++) {
        if(i == 0 && j == 0) {
          p_ehid_h_Ws_[i].push_back(p_ehid_h_W_);
          p_ehid_state_Ws_[i].push_back(p_ehid_state_W_);
          p_ehid_h_bs_[i].push_back(p_ehid_h_b_);
          p_e_ehid_Ws_[i].push_back(p_e_ehid_W_);
          p_e_ehid_bs_[i].push_back(p_e_ehid_b_);
        }else{
          p_ehid_h_Ws_[i].push_back(mod.add_parameters({(unsigned int)hidden_size_, (unsigned int)context_size_}));
          p_ehid_state_Ws_[i].push_back(mod.add_parameters({(unsigned int)hidden_size_, (unsigned int)state_size_}));
          p_ehid_h_bs_[i].push_back(mod.add_parameters({(unsigned int)hidden_size_, 1}));
          p_e_ehid_Ws_[i].push_back(mod.add_parameters({1, (unsigned int)hidden_size_}));
          p_e_ehid_bs_[i].push_back(mod.add_parameters({1, 1}));
        }
      }
    }
    cout << "Done" << endl;
  } else {
    THROW_ERROR("Illegal attention type: " << attention_type);
  }

                     
                   }


SeparateMultiTaskExternAttentional* SeparateMultiTaskExternAttentional::Read(std::istream & in, const vector<DictPtr> & vocab_src, const vector<DictPtr> & vocab_trg, vector<MultiTaskModelPtr> & mtmodels, dynet::Model & model) {
  int num_encoders, state_size;
  string version_id, attention_type, attention_hist = "none", lex_type = "none", line;
  int attention_context = 0;
  bool source_word_embedding_in_softmax = false;
  int source_word_embedding_in_softmax_context = 0;

  string encoder_type;

  if(!getline(in, line))
    THROW_ERROR("Premature end of model file when expecting ExternAttentional");
  istringstream iss(line);
  iss >> version_id;
  if (version_id == "separate_mt_extatt_005") {
    iss >> num_encoders >> attention_type >> attention_hist >> lex_type >> state_size >> attention_context >> source_word_embedding_in_softmax >> source_word_embedding_in_softmax_context >> encoder_type;
  } else {
    THROW_ERROR("Expecting a ExternAttentional of version mt_extatt_004, but got something different:" << endl << line);
  }

  vector<LinearEncoderPtr> encoders;
  if(encoder_type == SharedMultiTaskLinearEncoder::ModelID()) {
    while(num_encoders-- > 0) {
      SharedMultiTaskLinearEncoderPtr ptr(SharedMultiTaskLinearEncoder::Read(in, model));
      mtmodels.push_back(ptr);
      encoders.push_back(ptr);
    }
  }else if(encoder_type == SeparateMultiTaskLinearEncoder::ModelID()) {
    while(num_encoders-- > 0) {
      SeparateMultiTaskLinearEncoderPtr ptr(SeparateMultiTaskLinearEncoder::Read(in, model));
      mtmodels.push_back(ptr);
      encoders.push_back(ptr);
    }
  }else {
    THROW_ERROR("Unknown encoder type:" << encoder_type << endl << line);
  }
  return new SeparateMultiTaskExternAttentional(encoders, attention_type, attention_hist, state_size, lex_type, vocab_src, vocab_trg, attention_context,source_word_embedding_in_softmax, source_word_embedding_in_softmax_context, encoder_type,model);
}

void SeparateMultiTaskExternAttentional::Write(std::ostream & out) {
  out << "separate_mt_extatt_005 " << encoders_.size() << " " << attention_type_ << " " << attention_hist_ << " " << lex_type_ << " " << state_size_ << " " << attention_context_ << " " << source_word_embedding_in_softmax_ << " " << source_word_embedding_in_softmax_context_ << " " << encoder_type_ << endl;
  for(auto & enc : encoders_) enc->Write(out);
}


SeparateMultiTaskEncoderAttentional::SeparateMultiTaskEncoderAttentional(
           const ExternAttentionalPtr & extern_calc,
           const NeuralLMPtr & decoder, string decoder_type,int source_voc_size,int target_voc_size,
           dynet::Model & model)
  : EncoderAttentional(extern_calc,decoder,model),decoder_type_(decoder_type) {

  // Encoder to decoder mapping parameters
  int enc2dec_in = extern_calc->GetContextSize();
  int enc2dec_out = decoder_->GetNumLayers() * decoder_->GetNumNodes();

    for(int i = 0; i < source_voc_size; i++) {
      p_enc2dec_Ws_.push_back(vector<dynet::Parameter>());
      p_enc2dec_bs_.push_back(vector<dynet::Parameter>());
      for(int j = 0; j < target_voc_size; j++) {
        cout << "Create Attention Model:" << i << " " << j << endl;
        if(i == 0 && j == 0) {
          p_enc2dec_Ws_[i].push_back(p_enc2dec_W_);
          p_enc2dec_bs_[i].push_back(p_enc2dec_b_);
        }else{
          p_enc2dec_Ws_[i].push_back(model.add_parameters({(unsigned int)enc2dec_out, (unsigned int)enc2dec_in}));
          p_enc2dec_bs_[i].push_back(model.add_parameters({(unsigned int)enc2dec_out}));
        }
      }
    }
    cout << "Done" << endl;
    
    
  }


SeparateMultiTaskEncoderAttentional* SeparateMultiTaskEncoderAttentional::Read(const vector<DictPtr> & vocab_src, const vector<DictPtr> & vocab_trg, std::istream & in, vector<MultiTaskModelPtr> & mtmodels, dynet::Model & model) {
  string version_id, line;
  string decoder_type;
  int source_voc_size;
  int target_voc_size;
  if(!getline(in, line))
    THROW_ERROR("Premature end of model file when expecting SeparateMultiTaskEncoderAttentional");
  istringstream iss(line);
  iss >> version_id >> decoder_type >> source_voc_size >> target_voc_size;
  if(version_id != "separate_mt_encatt_001")
    THROW_ERROR("Expecting a SeparateMultiTaskEncoderAttentional of version mt_encatt_001, but got something different:" << endl << line);
  SeparateMultiTaskExternAttentionalPtr extern_calc(SeparateMultiTaskExternAttentional::Read(in, vocab_src, vocab_trg, mtmodels,model));
  NeuralLMPtr decoder;
  if(decoder_type == SharedMultiTaskNeuralLM::ModelID()) {
    SharedMultiTaskNeuralLM * slm = SharedMultiTaskNeuralLM::Read(vocab_trg, in, model);
    mtmodels.push_back(SharedMultiTaskNeuralLMPtr(slm));
    NeuralLM * lm = slm;
    decoder.reset(lm);
  }else if(decoder_type == SeparateMultiTaskNeuralLM::ModelID()) {
    SeparateMultiTaskNeuralLM * slm = SeparateMultiTaskNeuralLM::Read(vocab_trg, in, model);
    mtmodels.push_back(SeparateMultiTaskNeuralLMPtr(slm));
    NeuralLM * lm = slm;
    decoder.reset(lm);
  }else {
    THROW_ERROR("Unknown decoder type:" << decoder_type << endl << line);
  }
  decoder->SetAttention(extern_calc);
  return new SeparateMultiTaskEncoderAttentional(extern_calc, decoder, decoder_type,source_voc_size,target_voc_size,model);
}


void SeparateMultiTaskEncoderAttentional::Write(std::ostream & out) {
  out << "separate_mt_encatt_001" << " " << decoder_type_ << " " << p_enc2dec_Ws_.size() << " " << p_enc2dec_Ws_[0].size() << endl;
  extern_calc_->Write(out);
  decoder_->Write(out);
}
