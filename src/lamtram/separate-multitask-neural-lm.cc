#include <lamtram/separate-multitask-neural-lm.h>
#include <lamtram/macros.h>
#include <lamtram/builder-factory.h>
#include <lamtram/extern-calculator.h>
#include <lamtram/softmax-factory.h>
#include <lamtram/string-util.h>
#include <dynet/dict.h>
#include <dynet/model.h>
#include <dynet/nodes.h>
#include <dynet/rnn.h>
#include <boost/range/irange.hpp>
#include <lamtram/gru-cond.h>
#include <ctime>
#include <fstream>

using namespace std;
using namespace lamtram;

SeparateMultiTaskNeuralLM::SeparateMultiTaskNeuralLM(const vector<DictPtr> & vocabs, int ngram_context, int extern_context, bool extern_feed,
           int wordrep_size, const BuilderSpec & hidden_spec, int unk_id, const std::string & softmax_sig, bool word_embedding_in_softmax,
           int attention_context, bool source_word_embedding_in_softmax, int source_word_embedding_in_softmax_context,
           dynet::Model & model) : NeuralLM(vocabs[0],ngram_context,extern_context,extern_feed,wordrep_size,hidden_spec,unk_id,softmax_sig,word_embedding_in_softmax,
           attention_context,source_word_embedding_in_softmax,source_word_embedding_in_softmax_context,model) {

  // Word representations
  vocabs_ = vocabs;
  ps_wr_W_.push_back(p_wr_W_);
  softmaxes_.push_back(softmax_);
  builders_.push_back(builder_);
  for(int i = 1; i < vocabs_.size(); i++) {
    ps_wr_W_.push_back(model.add_lookup_parameters(vocabs[i]->size(), {(unsigned int)wordrep_size_})); 

    // Hidden layers
    builders_.push_back(BuilderFactory::CreateBuilder(hidden_spec_,
                       ngram_context*wordrep_size_ + (extern_feed ? extern_context : 0),
                       model));


    // Create the softmax
    int numberOfEncoder = 2;
    int softmax_input_size = hidden_spec_.nodes + extern_context;
    softmax_input_size += (word_embedding_in_softmax ? ngram_context*wordrep_size_ : 0);
    softmax_input_size += attention_context * 2 * extern_context;
    softmax_input_size += (source_word_embedding_in_softmax ? (source_word_embedding_in_softmax_context*2 +1) * wordrep_size_ * numberOfEncoder: 0);
    softmaxes_.push_back(SoftmaxFactory::CreateSoftmax(softmax_sig, softmax_input_size, vocabs[i], model));

  }

}

SeparateMultiTaskNeuralLM::SeparateMultiTaskNeuralLM(const vector<DictPtr> & vocabs, int ngram_context, int extern_context, bool extern_feed,
           int wordrep_size, const BuilderSpec & hidden_spec, int unk_id, const std::string & softmax_sig, bool word_embedding_in_softmax,
           int attention_context, bool source_word_embedding_in_softmax, int source_word_embedding_in_softmax_context,
           ExternCalculatorPtr & att, dynet::Model & model) : NeuralLM(vocabs[0],ngram_context,extern_context,extern_feed,wordrep_size,hidden_spec,unk_id,softmax_sig,word_embedding_in_softmax,
           attention_context,source_word_embedding_in_softmax,source_word_embedding_in_softmax_context,att,model) {

  // Word representations
  vocabs_ = vocabs;
  ps_wr_W_.push_back(p_wr_W_);
  softmaxes_.push_back(softmax_);
  builders_.push_back(builder_);
  cond_builders_.push_back(cond_builder_);
  for(int i = 1; i < vocabs_.size(); i++) {
    ps_wr_W_.push_back(model.add_lookup_parameters(vocabs[i]->size(), {(unsigned int)wordrep_size_})); 

    cond_builders_.push_back(BuilderFactory::CreateBuilder(hidden_spec_,
                       ngram_context_*wordrep_size_ , extern_context_,
                       model,att));
    builders_.push_back((BuilderPtr) cond_builders_[cond_builders_.size() -1]);
    // Create the softmax
    int numberOfEncoder = 2;
    int softmax_input_size = hidden_spec_.nodes + extern_context;
    softmax_input_size += (word_embedding_in_softmax ? ngram_context*wordrep_size_ : 0);
    softmax_input_size += attention_context * 2 * extern_context;
    softmax_input_size += (source_word_embedding_in_softmax ? (source_word_embedding_in_softmax_context*2 +1) * wordrep_size_ * numberOfEncoder: 0);
    softmaxes_.push_back(SoftmaxFactory::CreateSoftmax(softmax_sig, softmax_input_size, vocabs[i], model));

  }

}



SeparateMultiTaskNeuralLM* SeparateMultiTaskNeuralLM::Read(const vector<DictPtr> & vocabs, std::istream & in, dynet::Model & model) {
  string voc_sizes;
  int ngram_context, extern_context = 0, wordrep_size, unk_id;
  bool extern_feed;
  string version_id, hidden_spec, line, softmax_sig;
  bool word_embedding_in_softmax = false;
  bool intermediate_att = false;
  int attention_context = 0;
  bool source_word_embedding_in_softmax = false;
  int source_word_embedding_in_softmax_context = 0;
if(!getline(in, line))
    THROW_ERROR("Premature end of model file when expecting Neural LM");
  istringstream iss(line);
  iss >> version_id;
  if(version_id == "separatemultinlm_005") {
    iss >> voc_sizes >> ngram_context >> extern_context >> extern_feed >> wordrep_size >> hidden_spec >> unk_id >> softmax_sig;
  }else if(version_id == "separatemultinlm_006") {
    iss >> voc_sizes >> ngram_context >> extern_context >> extern_feed >> wordrep_size >> hidden_spec >> unk_id >> softmax_sig >> word_embedding_in_softmax;
  }else if(version_id == "separatemultinlm_007") {
    iss >> voc_sizes >> ngram_context >> extern_context >> extern_feed >> wordrep_size >> hidden_spec >> unk_id >> softmax_sig >> word_embedding_in_softmax >> intermediate_att;
  }else if(version_id == "separatemultinlm_008") {
    iss >> voc_sizes >> ngram_context >> extern_context >> extern_feed >> wordrep_size >> hidden_spec >> unk_id >> softmax_sig >> word_embedding_in_softmax >> intermediate_att >> attention_context >> source_word_embedding_in_softmax >> source_word_embedding_in_softmax_context;
  } else {
    THROW_ERROR("Expecting a Neural LM of version separatemultinlm_008, but got something different:" << endl << line);
  }
  vector<string> v = Tokenize(voc_sizes,"|");
  for(int i = 0; i < v.size(); i++) {assert(atoi(v[i].c_str()) == vocabs[i]->size());};
  if(intermediate_att) {
    std::shared_ptr<ExternCalculator> p;
    return new SeparateMultiTaskNeuralLM(vocabs, ngram_context, extern_context, extern_feed, wordrep_size, hidden_spec, unk_id, softmax_sig, word_embedding_in_softmax,
    attention_context,source_word_embedding_in_softmax,source_word_embedding_in_softmax_context,p,model);
  }else{
    return new SeparateMultiTaskNeuralLM(vocabs, ngram_context, extern_context, extern_feed, wordrep_size, hidden_spec, unk_id, softmax_sig, word_embedding_in_softmax,
    attention_context,source_word_embedding_in_softmax,source_word_embedding_in_softmax_context,model);
  }
    
}
void SeparateMultiTaskNeuralLM::Write(std::ostream & out) {
  out << "separatemultinlm_008 ";
  out << vocabs_[0]->size();
  for (int i = 1; i < vocabs_.size(); i++) {
    out << "|" << vocabs_[i]->size();
  }
  out << " " << ngram_context_ << " " << extern_context_ << " " << extern_feed_ << " " << wordrep_size_ << " " << hidden_spec_ << " " << unk_id_ << " " << softmax_->GetSig() << " " << word_embedding_in_softmax_ << " " << intermediate_att_ << " " << attention_context_ << " " << source_word_embedding_in_softmax_ << " " << source_word_embedding_in_softmax_context_ << " " << endl;
}

void SeparateMultiTaskNeuralLM::SetDropout(float dropout) { 
  for(auto& b : builders_) {
    b->set_dropout(dropout);
  }
  for(auto& s : softmaxes_) {
    s->SetDropout(dropout);
  }
  dropout_rate = dropout;
}

void SeparateMultiTaskNeuralLM::SetAttention(ExternCalculatorPtr att)
{ 
  if(hidden_spec_.type == "gru-cond" || hidden_spec_.type == "lstm-cond") {
    for(auto& builder: builders_) {
      GRUCONDBuilder * b = (GRUCONDBuilder *) builder.get();
      b->SetAttention(att);
    }
  }
}
