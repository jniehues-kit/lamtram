#include <lamtram/softmax-full.h>
#include <lamtram/macros.h>
#include <dynet/expr.h>
#include <dynet/dict.h>

using namespace lamtram;
using namespace dynet::expr;
using namespace std;

SoftmaxFull::SoftmaxFull(const std::string & sig, int input_size, const DictPtr & vocab, dynet::Model & mod) : SoftmaxBase(sig,input_size,vocab,mod) {
  p_sm_W_ = mod.add_parameters({(unsigned int)vocab->size(), (unsigned int)input_size});
  p_sm_b_ = mod.add_parameters({(unsigned int)vocab->size()}); 
  dropout_rate = 0.f;
}

void SoftmaxFull::NewGraph(dynet::ComputationGraph & cg) {
  i_sm_b_ = parameter(cg, p_sm_b_);
  i_sm_W_ = parameter(cg, p_sm_W_);
}

// Calculate training loss for one word
dynet::expr::Expression SoftmaxFull::CalcLoss(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ngram, bool train) {
  Expression input = in;
  if (dropout_rate) input = dropout(input, dropout_rate);
  Expression score = affine_transform({i_sm_b_, i_sm_W_, input});
  if(prior.pg != nullptr) score = score + prior;
  return pickneglogsoftmax(score, *ngram.rbegin());
}
// Calculate training loss for multiple words
dynet::expr::Expression SoftmaxFull::CalcLoss(dynet::expr::Expression & in, dynet::expr::Expression & prior, const std::vector<Sentence> & ngrams, bool train) {
  Expression input = in;
  if (dropout_rate) input = dropout(input, dropout_rate);
  Expression score = affine_transform({i_sm_b_, i_sm_W_, input});
  if(prior.pg != nullptr) score = score + prior;
  std::vector<unsigned> wvec(ngrams.size());
  for(size_t i = 0; i < ngrams.size(); i++)
    wvec[i] = *ngrams[i].rbegin();
  return pickneglogsoftmax(score, wvec);
}

// Calculate the full probability distribution
dynet::expr::Expression SoftmaxFull::CalcProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ctxt, bool train) {
  Expression input = in;
  if (dropout_rate) input = dropout(input, dropout_rate);
  return (prior.pg != nullptr ? 
          softmax(affine_transform({i_sm_b_, i_sm_W_, input}) + prior) :
          softmax(affine_transform({i_sm_b_, i_sm_W_, input})));
}
dynet::expr::Expression SoftmaxFull::CalcProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const vector<Sentence> & ctxt, bool train) {
  Expression input = in;
  if (dropout_rate) input = dropout(input, dropout_rate);
  return (prior.pg != nullptr ? 
          softmax(affine_transform({i_sm_b_, i_sm_W_, input}) + prior) :
          softmax(affine_transform({i_sm_b_, i_sm_W_, input})));
}
dynet::expr::Expression SoftmaxFull::CalcLogProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const Sentence & ctxt, bool train) {
  Expression input = in;
  if (dropout_rate) input = dropout(input, dropout_rate);
  return (prior.pg != nullptr ? 
          log_softmax(affine_transform({i_sm_b_, i_sm_W_, input}) + prior) :
          log_softmax(affine_transform({i_sm_b_, i_sm_W_, input})));
}
dynet::expr::Expression SoftmaxFull::CalcLogProb(dynet::expr::Expression & in, dynet::expr::Expression & prior, const vector<Sentence> & ctxt, bool train) {
  Expression input = in;
  if (dropout_rate) input = dropout(input, dropout_rate);
  return (prior.pg != nullptr ? 
          log_softmax(affine_transform({i_sm_b_, i_sm_W_, input}) + prior) :
          log_softmax(affine_transform({i_sm_b_, i_sm_W_, input})));
}

void SoftmaxFull::SetDropout(float dropout) { dropout_rate = dropout;}
