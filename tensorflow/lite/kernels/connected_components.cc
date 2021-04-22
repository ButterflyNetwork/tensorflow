//
// Created by Benjamin Sternlieb on 4/21/21.
//
#include <iostream>

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

  std::cout << "**** In Prepare ****" << std::endl;

  // Should only have 1 input and 1 output
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Retrieve input and output tensors.
  const TfLiteTensor *input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor *output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output));

  std::cout << "**** output type " << TfLiteTypeGetName(output->type) << " ****" << std::endl;


  // Run with just Float32 for now.
  // TODO: Should probably support int32 too?
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt64);

  TfLiteIntArray *output_size = TfLiteIntArrayCopy(input->dims);

  std::cout << "**** output_size 0 " << output_size->data[0] << std::endl;
  std::cout << "***  output_size 1 " << output_size->data[1] << std::endl;
  std::cout << "***  output_size 2 " << output_size->data[2] << std::endl;

  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  std::cout << "**** In Eval ****" << std::endl;

  TfLiteTensor *output;
  TF_LITE_ENSURE_OK(context,GetOutputSafe(context, node, kOutputTensor, &output));

  const TfLiteTensor *input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));

  auto *input_data = GetTensorData<float>(input);

  const int flat_size = GetTensorShape(output).FlatSize();

  std::cout << "**** flat_size " << flat_size << "  ****" << std::endl;
  std::cout << "**** output type " << TfLiteTypeGetName(output->type) << " ****" << std::endl;

  auto *output_data = GetTensorData<int32_t>(output);




  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = int32_t{i};
    std::cout << "**** in loop i = " << i << " Input Value = " << input_data[i] << std::endl;
    std::cout << "**** in loop i = " << i << " Output Value = " << output_data[i] << std::endl;
    std::cout << "**** sizeof = " << sizeof(output_data[i]) << std::endl;
  }
  return kTfLiteOk;
}
}  // namespace connected_component

TfLiteRegistration* Register_CONNECTED_COMPONENTS() {
  static TfLiteRegistration r = {
      /*init=*/nullptr, /*free=*/nullptr, connected_components::Prepare, connected_components::Eval
  };
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite