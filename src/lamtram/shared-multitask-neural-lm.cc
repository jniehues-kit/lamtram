#include <lamtram/shared-multitask-neural-lm.h>
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

SharedMultiTaskNeuralLM::SharedMultiTaskNeuralLM(const vector<DictPtr> & vocabs, int ngram_context, int extern_context, bool extern_feed,
           int wordrep_size, const BuilderSpec & hidden_spec, int unk_id, const std::string & softmax_sig, bool word_embedding_in_softmax,
           int attention_context, bool source_word_embedding_in_softmax, int source_word_embedding_in_softmax_context,
           dynet::Model & model) : NeuralLM(vocabs[0],ngram_context,extern_context,extern_feed,wordrep_size,hidden_spec,unk_id,softmax_sig,word_embedding_in_softmax,
           attention_context,source_word_embedding_in_softmax,source_word_embedding_in_softmax_context,model) {

  // Word representations
  vocabs_ = vocabs;
  ps_wr_W_.push_back(p_wr_W_);
  softmaxes_.push_back(softmax_);
  for(int i = 1; i < vocabs_.size(); i++) {
    ps_wr_W_.push_back(model.add_lookup_parameters(vocabs[i]->size(), {(unsigned int)wordrep_size})); 

    // Create the softmax
    int numberOfEncoder = 2;
    int softmax_input_size = hidden_spec_.nodes + extern_context;
    softmax_input_size += (word_embedding_in_softmax ? ngram_context*wordrep_size_ : 0);
    softmax_input_size += attention_context * 2 * extern_context;
    softmax_input_size += (source_word_embedding_in_softmax ? (source_word_embedding_in_softmax_context*2 +1) * wordrep_size_ * numberOfEncoder: 0);
    softmaxes_.push_back(SoftmaxFactory::CreateSoftmax(softmax_sig, softmax_input_size, vocabs[i], model));

  }

}

SharedMultiTaskNeuralLM::SharedMultiTaskNeuralLM(const vector<DictPtr> & vocabs, int ngram_context, int extern_context, bool extern_feed,
           int wordrep_size, const BuilderSpec & hidden_spec, int unk_id, const std::string & softmax_sig, bool word_embedding_in_softmax,
           int attention_context, bool source_word_embedding_in_softmax, int source_word_embedding_in_softmax_context,
           ExternCalculatorPtr & att, dynet::Model & model) : NeuralLM(vocabs[0],ngram_context,extern_context,extern_feed,wordrep_size,hidden_spec,unk_id,softmax_sig,word_embedding_in_softmax,
           attention_context,source_word_embedding_in_softmax,source_word_embedding_in_softmax_context,att,model) {

  // Word representations
  vocabs_ = vocabs;
  ps_wr_W_.push_back(p_wr_W_);
  softmaxes_.push_back(softmax_);
  for(int i = 1; i < vocabs_.size(); i++) {
    ps_wr_W_.push_back(model.add_lookup_parameters(vocabs[i]->size(), {(unsigned int)wordrep_size})); 

    // Create the softmax
    int numberOfEncoder = 2;
    int softmax_input_size = hidden_spec_.nodes + extern_context;
    softmax_input_size += (word_embedding_in_softmax ? ngram_context*wordrep_size_ : 0);
    softmax_input_size += attention_context * 2 * extern_context;
    softmax_input_size += (source_word_embedding_in_softmax ? (source_word_embedding_in_softmax_context*2 +1) * wordrep_size_ * numberOfEncoder: 0);
    softmaxes_.push_back(SoftmaxFactory::CreateSoftmax(softmax_sig, softmax_input_size, vocabs[i], model));

  }

}



SharedMultiTaskNeuralLM* SharedMultiTaskNeuralLM::Read(const vector<DictPtr> & vocabs, std::istream & in, dynet::Model & model) {
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
  if(version_id == "sharedmultinlm_005") {
    iss >> voc_sizes >> ngram_context >> extern_context >> extern_feed >> wordrep_size >> hidden_spec >> unk_id >> softmax_sig;
  }else if(version_id == "sharedmultinlm_006") {
    iss >> voc_sizes >> ngram_context >> extern_context >> extern_feed >> wordrep_size >> hidden_spec >> unk_id >> softmax_sig >> word_embedding_in_softmax;
  }else if(version_id == "sharedmultinlm_007") {
    iss >> voc_sizes >> ngram_context >> extern_context >> extern_feed >> wordrep_size >> hidden_spec >> unk_id >> softmax_sig >> word_embedding_in_softmax >> intermediate_att;
  }else if(version_id == "sharedmultinlm_008") {
    iss >> voc_sizes >> ngram_context >> extern_context >> extern_feed >> wordrep_size >> hidden_spec >> unk_id >> softmax_sig >> word_embedding_in_softmax >> intermediate_att >> attention_context >> source_word_embedding_in_softmax >> source_word_embedding_in_softmax_context;
  } else {
    THROW_ERROR("Expecting a Neural LM of version sharedmultinlm_008, but got something different:" << endl << line);
  }
  vector<string> v = Tokenize(voc_sizes,"|");
  for(int i = 0; i < v.size(); i++) {assert(atoi(v[i].c_str()) == vocabs[i]->size());};
  if(intermediate_att) {
    std::shared_ptr<ExternCalculator> p;
    return new SharedMultiTaskNeuralLM(vocabs, ngram_context, extern_context, extern_feed, wordrep_size, hidden_spec, unk_id, softmax_sig, word_embedding_in_softmax,
    attention_context,source_word_embedding_in_softmax,source_word_embedding_in_softmax_context,p,model);
  }else{
    return new SharedMultiTaskNeuralLM(vocabs, ngram_context, extern_context, extern_feed, wordrep_size, hidden_spec, unk_id, softmax_sig, word_embedding_in_softmax,
    attention_context,source_word_embedding_in_softmax,source_word_embedding_in_softmax_context,model);
  }
    
}
void SharedMultiTaskNeuralLM::Write(std::ostream & out) {
  out << "sharedmultinlm_008 ";
  out << vocabs_[0]->size();
  for (int i = 1; i < vocabs_.size(); i++) {
    out << "|" << vocabs_[i]->size();
  }
  out << " " << ngram_context_ << " " << extern_context_ << " " << extern_feed_ << " " << wordrep_size_ << " " << hidden_spec_ << " " << unk_id_ << " " << softmax_->GetSig() << " " << word_embedding_in_softmax_ << " " << intermediate_att_ << " " << attention_context_ << " " << source_word_embedding_in_softmax_ << " " << source_word_embedding_in_softmax_context_ << " " << endl;
}
