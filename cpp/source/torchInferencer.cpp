#include "torchInferencer.h"

TorchInference::TorchInference(const std::string& torchModelPath, const std::map<int, std::string>& modelClasses, const int imgW, const int imgH,
    const bool& runWithCuda, const float scoreThresh, const float nmsThresh, const int maxDet)
    : modelPath{ torchModelPath }, classes{ modelClasses }, cudaEnabled{ runWithCuda }, modelScoreThreshold{ scoreThresh },
    modelNMSThreshold{ nmsThresh }, modelMaxDet{ maxDet } {
    modelShape.width = imgW;
    modelShape.height = imgH;
}

std::vector<Detection> TorchInference::runInference(const cv::Mat& input) {
    cv::Mat modelInput = input;
    if (letterBoxForSquare && modelShape.width == modelShape.height)
        modelInput = formatToSquare(modelInput);

    torch::Device device(cudaEnabled ? torch::kCUDA : torch::kCPU);

    torch::Tensor image_tensor = torch::from_blob(modelInput.data, { modelInput.rows, modelInput.cols, 3 }, torch::kByte).to(device);
    image_tensor = image_tensor.toType(torch::kFloat32).div(255);
    image_tensor = image_tensor.permute({ 2, 0, 1 });
    image_tensor = image_tensor.unsqueeze(0);
    std::vector<torch::jit::IValue> inputs{ image_tensor };

    torch::Tensor output = net.forward(inputs).toTensor().cpu();

    auto keep = nonMaxSuppression(output)[0];
    auto boxes = keep.index({ Slice(), Slice(None, 4) });
    keep.index_put_({ Slice(), Slice(None, 4) }, scaleBoxes({ modelInput.rows, modelInput.cols }, boxes, { input.rows, input.cols }));

    std::vector<Detection> detections{};
    for (int i = 0; i < keep.size(0); ++i)
    {
        int x1 = keep[i][0].item().toFloat();
        int y1 = keep[i][1].item().toFloat();
        int x2 = keep[i][2].item().toFloat();
        int y2 = keep[i][3].item().toFloat();
        float conf = keep[i][4].item().toFloat();
        int cls = keep[i][5].item().toInt();

        Detection result;
        result.class_id = cls;
        result.confidence = conf;

        result.className = classes.at(result.class_id);
        result.x = x1;
        result.y = y1;
        result.w = x2 - x1;
        result.h = y2 - y1;

        detections.push_back(result);
    }

    return detections;
}

void TorchInference::loadTorchNetwork() {
    net = torch::jit::load(modelPath);
    net.eval();
    if (cudaEnabled && torch::cuda::is_available())
    {
        torch::Device device(torch::kCUDA);
        std::cout << "\nRunning on CUDA" << std::endl;
        net.to(device, torch::kFloat32);
        cudaEnabled = true;
    }
    else
    {
        torch::Device device(torch::kCPU);
        std::cout << "\nRunning on CPU" << std::endl;
        net.to(device, torch::kFloat32);
        cudaEnabled = false;
    }
}

cv::Mat TorchInference::formatToSquare(const cv::Mat& source) {
    if (source.cols == modelShape.width && source.rows == modelShape.height) {
        return source;
    }

    float resize_scale = generateScale(source);
    int new_shape_w = std::round(source.cols * resize_scale);
    int new_shape_h = std::round(source.rows * resize_scale);
    float padw = (modelShape.width - new_shape_w) / 2.;
    float padh = (modelShape.height - new_shape_h) / 2.;

    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::Mat result;
    cv::resize(source, result,
        cv::Size(new_shape_w, new_shape_h),
        0, 0, cv::INTER_AREA);

    cv::copyMakeBorder(result, result, top, bottom, left, right,
        cv::BORDER_CONSTANT, cv::Scalar(114.));
    return result;
}

float TorchInference::generateScale(const cv::Mat& image) {
    int origin_w = image.cols;
    int origin_h = image.rows;

    int target_h = modelShape.height;
    int target_w = modelShape.width;

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);
    return resize_scale;
}

torch::Tensor TorchInference::xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    auto dw = x.index({ "...", 2 }).div(2);
    auto dh = x.index({ "...", 3 }).div(2);
    y.index_put_({ "...", 0 }, x.index({ "...", 0 }) - dw);
    y.index_put_({ "...", 1 }, x.index({ "...", 1 }) - dh);
    y.index_put_({ "...", 2 }, x.index({ "...", 0 }) + dw);
    y.index_put_({ "...", 3 }, x.index({ "...", 1 }) + dh);
    return y;
}

torch::Tensor TorchInference::scaleBoxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape) {
    auto gain = (std::min)((float)img1_shape[0] / img0_shape[0], (float)img1_shape[1] / img0_shape[1]);
    auto pad0 = std::round((float)(img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
    auto pad1 = std::round((float)(img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);

    boxes.index_put_({ "...", 0 }, boxes.index({ "...", 0 }) - pad0);
    boxes.index_put_({ "...", 2 }, boxes.index({ "...", 2 }) - pad0);
    boxes.index_put_({ "...", 1 }, boxes.index({ "...", 1 }) - pad1);
    boxes.index_put_({ "...", 3 }, boxes.index({ "...", 3 }) - pad1);
    boxes.index_put_({ "...", Slice(None, 4) }, boxes.index({ "...", Slice(None, 4) }).div(gain));
    return boxes;
}

torch::Tensor TorchInference::nonMaxSuppression(torch::Tensor& prediction) {
    auto bs = prediction.size(0);
    auto nc = prediction.size(1) - 4;
    auto nm = prediction.size(1) - nc - 4;
    auto mi = 4 + nc;
    auto xc = prediction.index({ Slice(), Slice(4, mi) }).amax(1) > modelScoreThreshold;

    prediction = prediction.transpose(-1, -2);
    prediction.index_put_({ "...", Slice({None, 4}) }, xywh2xyxy(prediction.index({ "...", Slice(None, 4) })));

    std::vector<torch::Tensor> output;
    for (int i = 0; i < bs; i++) {
        output.push_back(torch::zeros({ 0, 6 + nm }, prediction.device()));
    }

    for (int xi = 0; xi < prediction.size(0); xi++) {
        auto x = prediction[xi];
        x = x.index({ xc[xi] });
        auto x_split = x.split({ 4, nc, nm }, 1);
        auto box = x_split[0], cls = x_split[1], mask = x_split[2];
        auto [conf, j] = cls.max(1, true);
        x = torch::cat({ box, conf, j.toType(torch::kFloat), mask }, 1);
        x = x.index({ conf.view(-1) > modelScoreThreshold });
        int n = x.size(0);
        if (!n) { continue; }

        // NMS
        auto c = x.index({ Slice(), Slice{5, 6} }) * 7680;
        auto boxes = x.index({ Slice(), Slice(None, 4) }) + c;
        auto scores = x.index({ Slice(), 4 });
        auto i = nms(boxes, scores, modelNMSThreshold);
        i = i.index({ Slice(None, modelMaxDet) });
        output[xi] = x.index({ i });
    }

    return torch::stack(output);
}

torch::Tensor TorchInference::nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold) {
    if (bboxes.numel() == 0)
        return torch::empty({ 0 }, bboxes.options().dtype(torch::kLong));

    auto x1_t = bboxes.select(1, 0).contiguous();
    auto y1_t = bboxes.select(1, 1).contiguous();
    auto x2_t = bboxes.select(1, 2).contiguous();
    auto y2_t = bboxes.select(1, 3).contiguous();

    torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

    auto order_t = std::get<1>(
        scores.sort(/*stable=*/true, /*dim=*/0, /*descending=*/true));

    auto ndets = bboxes.size(0);
    torch::Tensor suppressed_t = torch::zeros({ ndets }, bboxes.options().dtype(torch::kByte));
    torch::Tensor keep_t = torch::zeros({ ndets }, bboxes.options().dtype(torch::kLong));

    auto suppressed = suppressed_t.data_ptr<uint8_t>();
    auto keep = keep_t.data_ptr<int64_t>();
    auto order = order_t.data_ptr<int64_t>();
    auto x1 = x1_t.data_ptr<float>();
    auto y1 = y1_t.data_ptr<float>();
    auto x2 = x2_t.data_ptr<float>();
    auto y2 = y2_t.data_ptr<float>();
    auto areas = areas_t.data_ptr<float>();

    int64_t num_to_keep = 0;

    for (int64_t _i = 0; _i < ndets; _i++) {
        auto i = order[_i];
        if (suppressed[i] == 1)
            continue;
        keep[num_to_keep++] = i;
        auto ix1 = x1[i];
        auto iy1 = y1[i];
        auto ix2 = x2[i];
        auto iy2 = y2[i];
        auto iarea = areas[i];

        for (int64_t _j = _i + 1; _j < ndets; _j++) {
            auto j = order[_j];
            if (suppressed[j] == 1)
                continue;
            auto xx1 = std::max(ix1, x1[j]);
            auto yy1 = std::max(iy1, y1[j]);
            auto xx2 = std::min(ix2, x2[j]);
            auto yy2 = std::min(iy2, y2[j]);

            auto w = std::max(static_cast<float>(0), xx2 - xx1);
            auto h = std::max(static_cast<float>(0), yy2 - yy1);
            auto inter = w * h;
            auto ovr = inter / (iarea + areas[j] - inter);
            if (ovr > iou_threshold)
                suppressed[j] = 1;
        }
    }
    return keep_t.narrow(0, 0, num_to_keep);
}