#include <algorithm>
#include <vector>

#include "caffe/layers/relnu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLnULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    Dtype alpha = this->layer_param_.relnu_param().alpha();
    Dtype beta = this->layer_param_.relnu_param().beta();
    for (int i = 0; i < count; ++i) {
        top_data[i] = bottom_data[i] > Dtype(0) ? (beta * std::log(alpha * bottom_data[i] + 1)) : Dtype(0);
    }
}

template <typename Dtype>
void ReLnULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
    if (propagate_down[0]) {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const int count = bottom[0]->count();
        Dtype alpha = this->layer_param_.relnu_param().alpha();
        Dtype beta = this->layer_param_.relnu_param().beta();
        for (int i = 0; i < count; ++i) {
            bottom_diff[i] = top_diff[i] * (bottom_data[i] > Dtype(0) ? (beta / (bottom_data[i] + Dtype(1) / alpha)) : Dtype(0));
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(ReLnULayer);
#endif

INSTANTIATE_CLASS(ReLnULayer);
REGISTER_LAYER_CLASS(ReLnU);

} // namespace caffe
