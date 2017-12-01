//
// Created by admins on 17-10-8.
//

#ifndef CAFFE_FLATPATCH_H_
#define CAFFE_FLATPATCH_H_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

    template <typename Dtype>
    class FlatPatchLayer : public Layer<Dtype> {
    public:
        explicit FlatPatchLayer(const LayerParameter &param) : Layer<Dtype>(param) {}

        virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top);

        virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

        virtual inline const char *type() const { return "FlatPatch"; }

        virtual inline int ExactNumBottomBlobs() const { return 1; }

        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down,
                                  const vector<Blob<Dtype> *> &bottom);

        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down,
                                  const vector<Blob<Dtype> *> &bottom);

        int size_h;
        int size_w;
    };

}  // namespace caffe

#endif //CAFFE_FLATPATCH_H_
