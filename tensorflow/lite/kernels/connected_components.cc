// This is a port of the connected_components module from
// tensorflow_addons. Input and output shapes are the same.
// We expect a shape of (N, H, W). The code supports both
// integer and float types. Output type is int64.
//
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace connected_components {

namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

constexpr int kTempTensorCount = 2;

}  // namespace

// Struct to hold temporary Tensors
struct OpData {
  int scratch_tensor_index;
};

// This is where we add the forest and rank tensors used by the
// addons::ImageConnectedCompontes OpKernel
void* Init(TfLiteContext* context, const char*, size_t) {
  auto* op_data = new OpData();
  context->AddTensors(context, kTempTensorCount, &op_data->scratch_tensor_index);
  return op_data;
}

// Clean up our OpData structure.
// TODO: Where are the tensors freed?
void Free(TfLiteContext*, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

// Prepare is called during the allocation of tensors.
// Here we
//  1: check on the number of inputs and outputs
//  2: check that we support the input and output types.
//  3: resize tensors based on the input shape.
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *input_t;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input_t));
  TfLiteTensor *output_t;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output_t));



  TF_LITE_ENSURE_EQ(context, input_t->dims->size, 3);
  TF_LITE_ENSURE_TYPES_EQ(context, input_t->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, output_t->type, kTfLiteInt64);

  // allocate temporary tensors
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(kTempTensorCount);

  // forest_t
  node->temporaries->data[0] = op_data->scratch_tensor_index;
  TfLiteTensor* forest_t;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/0, &forest_t));
  forest_t->type = kTfLiteInt64;
  forest_t->allocation_type = kTfLiteArenaRw;
  if (!TfLiteIntArrayEqual(forest_t->dims, input_t->dims)) {
    TfLiteIntArray* input_size = TfLiteIntArrayCopy(input_t->dims);
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, forest_t, input_size));
  }

  // rank_t
  node->temporaries->data[1] = op_data->scratch_tensor_index+1;
  TfLiteTensor* rank_t;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/1, &rank_t));
  rank_t->type = kTfLiteInt64;
  rank_t->allocation_type = kTfLiteArenaRw;
  if (!TfLiteIntArrayEqual(rank_t->dims, input_t->dims)) {
    TfLiteIntArray* input_size = TfLiteIntArrayCopy(input_t->dims);
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, rank_t, input_size));
  }


  if(!TfLiteIntArrayEqual(output_t->dims, input_t->dims)) {
    TfLiteIntArray *input_size = TfLiteIntArrayCopy(input_t->dims);
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_t, input_size));
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  TfLiteTensor *output_t;
  TF_LITE_ENSURE_OK(context,GetOutputSafe(context, node, kOutputTensor, &output_t));
  const TfLiteTensor *input_t;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input_t));
  const int flat_size = GetTensorShape(output_t).FlatSize();
  auto *output_data = GetTensorData<int64_t>(output_t);

  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = int64_t{i};
  }
  return kTfLiteOk;
}
}  // namespace connected_component

TfLiteRegistration* Register_CONNECTED_COMPONENTS() {
  static TfLiteRegistration r = {
      connected_components::Init,
      connected_components::Free,
      connected_components::Prepare,
      connected_components::Eval
  };
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite