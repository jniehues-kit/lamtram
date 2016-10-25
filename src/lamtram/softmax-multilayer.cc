#include <lamtram/softmax-multilayer.h>
#include <lamtram/macros.h>
#include <dynet/expr.h>
#include <dynet/dict.h>

using namespace lamtram;
using namespace dynet::expr;
using namespace std;

SoftmaxMultiLayer::SoftmaxMultiLayer(const std::string & sig, int input_size, const DictPtr & vocab, dynet::Model & mod) : SoftmaxBase(sig,input_size,vocab,mod) {

  vector<string> strs;
    boost::algorithm::split(strs, sig, boost::is_any_of(":"));

  int hiddenSize = stoi(strs[1]);
  if(hiddenSize <= 0) hiddenSize = GlobalVars::layer_size;
  p_sm_W_ = mod.add_parameters({(unsigned int)hiddenSize, (unsigned int)input_size});
  p_sm_b_ = mod.add_parameters({(unsigned int)hiddenSize});
  strs.erase(strs.begin(), strs.begin() + 2);
  std::string new_sig = boost::algorithm::join(strs, ":");
  softmax_ = SoftmaxFactory::CreateSoftmax(new_sig, hiddenSize, vocab, mod);
  dropout_rate = 0.f;
}

void SoftmaxMultiLayer::NewGraph(dynet::ComputationGraph & cg) {
  i_sm_b_ = parameter(cg, p_sm_b_);
  i_sm_W_ = parameter(cg, p_sm_W_);
  softmax_->NewGraph(cg);
}

// Calculate training loss for one word
dynet::expr::Expression SoftmaxMultiLayer::CalcLoss(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ngram, bool train) {
  Expression input = in;
  if (dropout_rate) input = dropout(input, dropout_rate);
  Expression score = affine_transform({i_sm_b_, i_sm_W_, input});
  score = tanh(score);
  return softmax_->CalcLoss(score,prior,ngram,train);
}
// Calculate training loss for multiple words
dynet::expr::Expression SoftmaxMultiLayer::CalcLoss(dynet::expr::Expression & in, dynet::expr::Expression & prior, const std::vector<Sentence> & ngrams, bool train) {
  Expression input = in;
  if (dropout_rate) input = dropout(input, dropout_rate);
  Expression score = affine_transform({i_sm_b_, i_sm_W_, input});
  score = tanh(score);
  return softmax_->CalcLoss(score,prior,ngrams,train);
}

// Calculate the full probability distribution
dynet::expr::Expression SoftmaxMultiLayer::CalcProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ctxt, bool train) {
  Expression input = in;
  if (dropout_rate) input = dropout(input, dropout_rate);
  dynet::expr::Expression h = tanh(affine_transform({i_sm_b_, i_sm_W_, input}));
  return softmax_->CalcProb(h,prior,ctxt,train);
}
dynet::expr::Expression SoftmaxMultiLayer::CalcProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const vector<Sentence> & ctxt, bool train) {
  Expression input = in;
  if (dropout_rate) input = dropout(input, dropout_rate);
  dynet::expr::Expression h = tanh(affine_transform({i_sm_b_, i_sm_W_, input}));
  return softmax_->CalcProb(h,prior,ctxt,train);
}
dynet::expr::Expression SoftmaxMultiLayer::CalcLogProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ctxt, bool train) {
  Expression input = in;
  if (dropout_rate) input = dropout(input, dropout_rate);
  dynet::expr::Expression h = tanh(affine_transform({i_sm_b_, i_sm_W_, input}));
  return softmax_->CalcLogProb(h,prior,ctxt,train);
}
dynet::expr::Expression SoftmaxMultiLayer::CalcLogProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const vector<Sentence> & ctxt, bool train) {
  Expression input = in;
  if (dropout_rate) input = dropout(input, dropout_rate);
  dynet::expr::Expression h = tanh(affine_transform({i_sm_b_, i_sm_W_, input}));
  return softmax_->CalcLogProb(h,prior,ctxt,train);
}

void SoftmaxMultiLayer::SetDropout(float dropout) { softmax_->SetDropout(dropout);dropout_rate = dropout;}
