#pragma once


namespace lamtram {



class MultiTaskModel {

public:

    virtual void SetVocabulary(int sourceIndex,int targetIndex) = 0;

};


typedef std::shared_ptr<MultiTaskModel> MultiTaskModelPtr;

}