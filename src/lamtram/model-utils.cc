
#include <lamtram/model-utils.h>
#include <lamtram/macros.h>
#include <lamtram/encoder-decoder.h>
#include <lamtram/encoder-attentional.h>
#include <lamtram/encoder-classifier.h>
#include <lamtram/multitask-encoder-attentional.h>
#include <lamtram/separate-multitask-encoder-attentional.h>
#include <lamtram/neural-lm.h>
#include <dynet/model.h>
#include <dynet/dict.h>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <fstream>

using namespace std;
using namespace lamtram;

void ModelUtils::WriteModelText(ostream & out, const dynet::Model & mod) {
    boost::archive::text_oarchive oa(out);
    oa << mod;
}
void ModelUtils::ReadModelText(istream & in, dynet::Model & mod) {
    boost::archive::text_iarchive ia(in);
    ia >> mod;
}


// Load a model from a stream
// Will return a pointer to the model, and reset the passed shared pointers
// with dynet::Model, and input, output vocabularies (if necessary)
template <class ModelType>
ModelType* ModelUtils::LoadBilingualModel(std::istream & model_in,
                                          std::shared_ptr<dynet::Model> & mod,
                                          DictPtr & vocab_src, DictPtr & vocab_trg) {
    vocab_src.reset(ReadDict(model_in));
    vocab_trg.reset(ReadDict(model_in));
    mod.reset(new dynet::Model);
    ModelType* ret = ModelType::Read(vocab_src, vocab_trg, model_in, *mod);
    ModelUtils::ReadModelText(model_in, *mod);
    return ret;
}

// Load a model from a text file
// Will return a pointer to the model, and reset the passed shared pointers
// with dynet::Model, and input, output vocabularies (if necessary)
template <class ModelType>
ModelType* ModelUtils::LoadBilingualModel(const std::string & file,
                                          std::shared_ptr<dynet::Model> & mod,
                                          DictPtr & vocab_src, DictPtr & vocab_trg) {
    ifstream model_in(file);
    if(!model_in) THROW_ERROR("Could not open model file " << file);
    return ModelUtils::LoadBilingualModel<ModelType>(model_in, mod, vocab_src, vocab_trg);
}


// Load a model from a stream
// Will return a pointer to the model, and reset the passed shared pointers
// with dynet::Model, and input, output vocabularies (if necessary)
template <class ModelType>
ModelType* ModelUtils::LoadMultitaskModel(std::istream & model_in,
                                          std::shared_ptr<dynet::Model> & mod,
                                          vector<DictPtr> & vocab_src, vector<DictPtr> & vocab_trg,
                                          vector<MultiTaskModelPtr> & mtmodels) {
    int src_voc_size,trg_voc_size;
    model_in >> src_voc_size >> trg_voc_size;
    //finish line before reading for dict
    string s;
    getline(model_in,s);
    //model_in >> s;
    //cout << s << endl;
    for(int i = 0; i < src_voc_size;i++) {
        vocab_src.push_back(std::shared_ptr<Dict>(ReadDict(model_in)));
    }
    for(int i = 0; i < trg_voc_size;i++) {
        vocab_trg.push_back(std::shared_ptr<Dict>(ReadDict(model_in)));
    }
    mod.reset(new dynet::Model);
    ModelType* ret = ModelType::Read(vocab_src, vocab_trg, model_in, mtmodels,*mod);
    ModelUtils::ReadModelText(model_in, *mod);
    return ret;
}

// Load a model from a text file
// Will return a pointer to the model, and reset the passed shared pointers
// with dynet::Model, and input, output vocabularies (if necessary)
template <class ModelType>
ModelType* ModelUtils::LoadMultitaskModel(const std::string & file,
                                          std::shared_ptr<dynet::Model> & mod,
                                          vector<DictPtr> & vocab_src, vector<DictPtr> & vocab_trg,
                                          vector<MultiTaskModelPtr> & mtmodels) {
    ifstream model_in(file);
    if(!model_in) THROW_ERROR("Could not open model file " << file);
    return ModelUtils::LoadMultitaskModel<ModelType>(model_in, mod, vocab_src, vocab_trg,mtmodels);
}



// Load a model from a stream
// Will return a pointer to the model, and reset the passed shared pointers
// with dynet::Model, and input, output vocabularies (if necessary)
template <class ModelType>
ModelType* ModelUtils::LoadMonolingualModel(std::istream & model_in,
                                          std::shared_ptr<dynet::Model> & mod,
                                          DictPtr & vocab_trg) {
    vocab_trg.reset(ReadDict(model_in));
    mod.reset(new dynet::Model);
    ModelType* ret = ModelType::Read(vocab_trg, model_in, *mod);
    ModelUtils::ReadModelText(model_in, *mod);
    return ret;
}

// Load a model from a text file
// Will return a pointer to the model, and reset the passed shared pointers
// with dynet::Model, and input, output vocabularies (if necessary)
template <class ModelType>
ModelType* ModelUtils::LoadMonolingualModel(const std::string & file,
                                          std::shared_ptr<dynet::Model> & mod,
                                          DictPtr & vocab_trg) {
    ifstream model_in(file);
    if(!model_in) THROW_ERROR("Could not open model file " << file);
    return ModelUtils::LoadMonolingualModel<ModelType>(model_in, mod, vocab_trg);
}

// Instantiate LoadModel
template
MultiTaskEncoderAttentional* ModelUtils::LoadMultitaskModel<MultiTaskEncoderAttentional>(std::istream & model_in,
                                                              std::shared_ptr<dynet::Model> & mod,
                                                              vector<DictPtr> & vocab_src, vector<DictPtr> & vocab_trg,
                                                              std::vector<MultiTaskModelPtr> & mtmodels);

template
MultiTaskEncoderAttentional* ModelUtils::LoadMultitaskModel<MultiTaskEncoderAttentional>(const std::string & infile,
                                                              std::shared_ptr<dynet::Model> & mod,
                                                              vector<DictPtr> & vocab_src, vector<DictPtr> & vocab_trg,
                                                              std::vector<MultiTaskModelPtr> & mtmodels);

template
SeparateMultiTaskEncoderAttentional* ModelUtils::LoadMultitaskModel<SeparateMultiTaskEncoderAttentional>(std::istream & model_in,
                                                              std::shared_ptr<dynet::Model> & mod,
                                                              vector<DictPtr> & vocab_src, vector<DictPtr> & vocab_trg,
                                                              std::vector<MultiTaskModelPtr> & mtmodels);

template
SeparateMultiTaskEncoderAttentional* ModelUtils::LoadMultitaskModel<SeparateMultiTaskEncoderAttentional>(const std::string & infile,
                                                              std::shared_ptr<dynet::Model> & mod,
                                                              vector<DictPtr> & vocab_src, vector<DictPtr> & vocab_trg,
                                                              std::vector<MultiTaskModelPtr> & mtmodels);


template
EncoderDecoder* ModelUtils::LoadBilingualModel<EncoderDecoder>(std::istream & model_in,
                                                      std::shared_ptr<dynet::Model> & mod,
                                                      DictPtr & vocab_src, DictPtr & vocab_trg);
template
EncoderAttentional* ModelUtils::LoadBilingualModel<EncoderAttentional>(std::istream & model_in,
                                                              std::shared_ptr<dynet::Model> & mod,
                                                              DictPtr & vocab_src, DictPtr & vocab_trg);
template
EncoderClassifier* ModelUtils::LoadBilingualModel<EncoderClassifier>(std::istream & model_in,
                                                            std::shared_ptr<dynet::Model> & mod,
                                                            DictPtr & vocab_src, DictPtr & vocab_trg);
template
NeuralLM* ModelUtils::LoadMonolingualModel<NeuralLM>(std::istream & model_in,
                                          std::shared_ptr<dynet::Model> & mod,
                                          DictPtr & vocab_trg);
template
EncoderDecoder* ModelUtils::LoadBilingualModel<EncoderDecoder>(const std::string & infile,
                                                      std::shared_ptr<dynet::Model> & mod,
                                                      DictPtr & vocab_src, DictPtr & vocab_trg);
template
EncoderAttentional* ModelUtils::LoadBilingualModel<EncoderAttentional>(const std::string & infile,
                                                              std::shared_ptr<dynet::Model> & mod,
                                                              DictPtr & vocab_src, DictPtr & vocab_trg);
template
EncoderClassifier* ModelUtils::LoadBilingualModel<EncoderClassifier>(const std::string & infile,
                                                            std::shared_ptr<dynet::Model> & mod,
                                                            DictPtr & vocab_src, DictPtr & vocab_trg);
template
NeuralLM* ModelUtils::LoadMonolingualModel<NeuralLM>(const std::string & infile,
                                                     std::shared_ptr<dynet::Model> & mod,
                                                     DictPtr & vocab_trg);
