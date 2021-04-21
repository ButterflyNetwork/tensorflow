//
// Created by Benjamin Sternlieb on 4/21/21.
//
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace connected_components {

namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

enum KernelType {
  kReference,
  kGenericOptimized,
};

}  // namespace

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {

  // Should only have 1 input and 1 output
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Retrieve input and output tensors.
  const TfLiteTensor *input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor *output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output));

  // Run with just Float32 for now.
  // TODO: Should probably support int32 too?
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);

  // output is always Int64
  output->type = kTfLiteInt64

  TfLiteIntArray *output_size = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  TfLiteTensor *output;
  TF_LITE_ENSURE_OK(context,GetOutputSafe(context, node, kOutputTensor, &output));

  const int flat_size = GetTensorShape(output).FlatSize();

  auto *output_data = GetTensorData<int64_t>(output)

  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = 1;
  }
  return kTfLiteOk;
}
}  // namespace connected_component

TfLiteRegistration* Register_CONNECTED_COMPONENTS() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                          fill::Prepare, fill::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite