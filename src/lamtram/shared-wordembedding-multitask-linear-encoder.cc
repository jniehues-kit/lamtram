#include <lamtram/separate-multitask-linear-encoder.h>
#include <lamtram/macros.h>
#include <lamtram/builder-factory.h>
#include <lamtram/string-util.h>
#include <dynet/model.h>
#include <dynet/nodes.h>
#include <dynet/rnn.h>
#include <boost/range/irange.hpp>
#include <ctime>
#include <fstream>

using namespace std;
using namespace lamtram;

SharedWordEmbeddingMultiTaskLinearEncoder::SharedWordEmbeddingMultiTaskLinearEncoder(const vector<int> & vocab_size, int wordrep_size,
           const BuilderSpec & hidden_spec, int unk_id,
           dynet::Model & model) :LinearEncoder(hidden_spec){
  wordrep_size_ = wordrep_size;
  unk_id_ = unk_id;
  reverse_ = false;
  for(int i = 0; i < vocab_size.size(); i++) {
    // Hidden layers
    builders_.push_back(BuilderFactory::CreateBuilder(hidden_spec_, wordrep_size, model));
    // Word representations
    vocab_sizes_.push_back(vocab_size[i]);
    ps_wr_W_.push_back(model.add_lookup_parameters(vocab_size[i], {(unsigned int)wordrep_size})); 
  }
  current_voc_ = 0;
}


SharedWordEmbeddingMultiTaskLinearEncoder* SharedWordEmbeddingMultiTaskLinearEncoder::Read(std::istream & in, dynet::Model & model) {
  string vocab_sizes;
  int wordrep_size, unk_id;
  string version_id, hidden_spec, line, reverse;
  if(!getline(in, line))
    THROW_ERROR("Premature end of model file when expecting Neural LM");
  istringstream iss(line);
  iss >> version_id >> vocab_sizes >> wordrep_size >> hidden_spec >> unk_id >> reverse;
  if(version_id != "separatemultilinenc_001")
    THROW_ERROR("Expecting a Neural LM of version linenc_001, but got something different:" << endl << line);
  vector<string> v = Tokenize(vocab_sizes,"|");
  vector<int> v_sizes;
  for(int i = 0; i < v.size(); i++) {v_sizes.push_back(atoi(v[i].c_str()));};
  SharedWordEmbeddingMultiTaskLinearEncoder * ret = new SharedWordEmbeddingMultiTaskLinearEncoder(v_sizes, wordrep_size, BuilderSpec(hidden_spec), unk_id, model);
  if(reverse == "rev") ret->SetReverse(true);
  return ret;
}
void SharedWordEmbeddingMultiTaskLinearEncoder::Write(std::ostream & out) {
  out << "separatemultilinenc_001 ";
  out << vocab_sizes_[0];
  for (int i = 1; i < vocab_sizes_.size(); i++) {
    out << "|" << vocab_sizes_[i];
  }
  out << " " << wordrep_size_ << " " << hidden_spec_ << " " << unk_id_ << " " << (reverse_?"rev":"for") << endl;
}

void SharedWordEmbeddingMultiTaskLinearEncoder::SetDropout(float dropout) { 
  for (auto& b : builders_) {
    b->set_dropout(dropout);
  }
}
    