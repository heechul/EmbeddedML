#include "NeuralNetwork.h"
#include "model.h"
#include "model_config.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Arduino.h>
#include <esp_heap_caps.h>
#include <math.h>

namespace {
constexpr size_t kTensorArenaSize = 256 * 1024;
bool validTensor(const TfLiteTensor* tensor, int height, int width, int channels,
                 float scale, int zero_point) {
    return tensor && tensor->dims && tensor->dims->size == 4 &&
        tensor->dims->data[0] == 1 && tensor->dims->data[1] == height &&
        tensor->dims->data[2] == width && tensor->dims->data[3] == channels &&
        tensor->type == kTfLiteInt8 && tensor->bytes == height * width * channels &&
        tensor->params.scale > 0 && fabsf(tensor->params.scale - scale) < 1e-7f &&
        tensor->params.zero_point == zero_point;
}
}

bool NeuralNetwork::begin() {
    if (ready_) return true;
    // Initialized once by setup(); the arena/interpreter live for the whole boot.
    const tflite::Model* model = tflite::GetModel(gmodel);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Unsupported TFLite schema");
        return false;
    }
    // Matches the notebook's integer-only Conv2D + Softmax model.
    static tflite::MicroMutableOpResolver<2> resolver;
    if (resolver.AddConv2D() != kTfLiteOk || resolver.AddSoftmax() != kTfLiteOk) {
        Serial.println("Operator registration failed");
        return false;
    }
    arena_ = static_cast<uint8_t*>(heap_caps_malloc(
        kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!arena_) {
        Serial.println("Tensor arena allocation failed; verify PSRAM is enabled");
        return false;
    }
    static tflite::MicroInterpreter interpreter(model, resolver, arena_, kTensorArenaSize);
    interpreter_ = &interpreter;
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors failed");
        return false;
    }
    input_ = interpreter_->input(0);
    output_ = interpreter_->output(0);
    if (!validTensor(input_, fomo::kInputHeight, fomo::kInputWidth, fomo::kChannels,
                     fomo::kInputScale, fomo::kInputZeroPoint) ||
        !validTensor(output_, fomo::kGridHeight, fomo::kGridWidth, fomo::kClassCount,
                     fomo::kOutputScale, fomo::kOutputZeroPoint)) {
        Serial.println("FOMO tensor contract mismatch; export all model files together");
        return false;
    }
    Serial.printf("FOMO model: %u bytes, tensor arena: %u / %u bytes\n",
        gmodel_len, static_cast<unsigned>(interpreter_->arena_used_bytes()),
        static_cast<unsigned>(kTensorArenaSize));
    ready_ = true;
    return true;
}

TfLiteStatus NeuralNetwork::predict() {
    return ready_ ? interpreter_->Invoke() : kTfLiteError;
}
