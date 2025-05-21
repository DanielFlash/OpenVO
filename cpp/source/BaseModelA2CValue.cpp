#include "BaseModelA2CValue.h"

BaseModelA2CValue::BaseModelA2CValue(int64_t inputSize, const std::vector<int64_t>& hiddenSizes)
    : m_inputSize{ inputSize }, m_hiddenSizes{ hiddenSizes } {
    int idx = 1;
    int64_t prevLayerSize = m_inputSize;
    for (const int64_t layerSize : m_hiddenSizes) {
        torch::nn::Linear fc(prevLayerSize, layerSize);
        m_fcLayers.push_back(fc);
        register_module("fc" + std::to_string(idx), fc);
        idx += 1;
        prevLayerSize = layerSize;
    }
    torch::nn::Linear lastLayer(prevLayerSize, 1);
    m_fcLayers.push_back(lastLayer);
    register_module("fc" + std::to_string(idx), lastLayer);
}

torch::Tensor BaseModelA2CValue::forward(torch::Tensor x) {
    int i = 0;
    for (; i < m_fcLayers.size() - 1; i++) {
        x = torch::nn::functional::relu(m_fcLayers[i]->forward(x));
    }
    return m_fcLayers[i]->forward(x);
}