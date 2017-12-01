#ifndef CAFFE_SWITCH_LAYER_HPP_
#define CAFFE_SWITCH_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * \ingroup ttic
 * @brief
 *
 * @author Liang Kang
 */
template <typename Dtype>
class SwitchLayer : public Layer<Dtype> {
public:
    explicit SwitchLayer(const LayerParameter& param)
        : Layer<Dtype>(param)
    {
    }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Switch"; }
    virtual inline int MinBottomBlobs() const { return 2; }
    virtual inline int MinTopBlobs() const { return 2; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    unsigned int BLOB_SIZE;
    bool stochastic;
    std::vector<Dtype> switch_path;
    std::vector<unsigned int> switch_map;
};

} // namespace caffe

#endif // CAFFE_SWITCH_LAYER_HPP_
