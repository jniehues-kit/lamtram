#include <lamtram/multitask-encoder-attentional.h>
#include <lamtram/shared-multitask-neural-lm.h>
#include <lamtram/shared-multitask-linear-encoder.h>
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


MultiTaskExternAttentional::MultiTaskExternAttentional(const std::vector<LinearEncoderPtr> & encoders,
                   const std::string & attention_type, const std::string & attention_hist, int state_size,
                   const std::string & lex_type,
                   const vector<DictPtr> & vocab_src, const vector<DictPtr> & vocab_trg,
                   int attention_context, bool source_word_embedding_in_softmax, int source_word_embedding_in_softmax_context, string encoder_type,
                   dynet::Model & mod): ExternAttentional(encoders,attention_type,attention_hist,state_size,lex_type,vocab_src[0],vocab_trg[0],attention_context,source_word_embedding_in_softmax,source_word_embedding_in_softmax_context,mod),encoder_type_(encoder_type) {
                     
                     
                   }


MultiTaskExternAttentional* MultiTaskExternAttentional::Read(std::istream & in, const vector<DictPtr> & vocab_src, const vector<DictPtr> & vocab_trg, vector<MultiTaskModelPtr> & mtmodels, dynet::Model & model) {
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
  if (version_id == "mt_extatt_005") {
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
  }else {
    THROW_ERROR("Unknown encoder type:" << encoder_type << endl << line);
  }
  return new MultiTaskExternAttentional(encoders, attention_type, attention_hist, state_size, lex_type, vocab_src, vocab_trg, attention_context,source_word_embedding_in_softmax, source_word_embedding_in_softmax_context, encoder_type,model);
}

void MultiTaskExternAttentional::Write(std::ostream & out) {
  out << "mt_extatt_005 " << encoders_.size() << " " << attention_type_ << " " << attention_hist_ << " " << lex_type_ << " " << state_size_ << " " << attention_context_ << " " << source_word_embedding_in_softmax_ << " " << source_word_embedding_in_softmax_context_ << " " << encoder_type_ << endl;
  for(auto & enc : encoders_) enc->Write(out);
}


MultiTaskEncoderAttentional::MultiTaskEncoderAttentional(
           const ExternAttentionalPtr & extern_calc,
           const NeuralLMPtr & decoder, string decoder_type,
           dynet::Model & model)
  : EncoderAttentional(extern_calc,decoder,model),decoder_type_(decoder_type) {
}


MultiTaskEncoderAttentional* MultiTaskEncoderAttentional::Read(const vector<DictPtr> & vocab_src, const vector<DictPtr> & vocab_trg, std::istream & in, vector<MultiTaskModelPtr> & mtmodels, dynet::Model & model) {
  string version_id, line;
  string decoder_type;
  if(!getline(in, line))
    THROW_ERROR("Premature end of model file when expecting MultiTaskEncoderAttentional");
  istringstream iss(line);
  iss >> version_id >> decoder_type;
  if(version_id != "mt_encatt_001")
    THROW_ERROR("Expecting a MultiTaskEncoderAttentional of version mt_encatt_001, but got something different:" << endl << line);
  MultiTaskExternAttentionalPtr extern_calc(MultiTaskExternAttentional::Read(in, vocab_src, vocab_trg, mtmodels,model));
  NeuralLMPtr decoder;
  if(decoder_type == SharedMultiTaskNeuralLM::ModelID()) {
    SharedMultiTaskNeuralLM * slm = SharedMultiTaskNeuralLM::Read(vocab_trg, in, model);
    mtmodels.push_back(SharedMultiTaskNeuralLMPtr(slm));
    NeuralLM * lm = slm;
    decoder.reset(lm);
  }else {
    THROW_ERROR("Unknown decoder type:" << decoder_type << endl << line);
  }
  decoder->SetAttention(extern_calc);
  return new MultiTaskEncoderAttentional(extern_calc, decoder, decoder_type,model);
}


void MultiTaskEncoderAttentional::Write(std::ostream & out) {
  out << "mt_encatt_001" << " " << decoder_type_ << endl;
  extern_calc_->Write(out);
  decoder_->Write(out);
}
