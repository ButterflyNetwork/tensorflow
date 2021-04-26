//
// Created by Benjamin Sternlieb on 4/21/21.
//
#include <iostream>
#include <iomanip>
#include <vector>

#include "tensorflow/lite/c/c_api.h"

template <class T>
void dump(const std::vector<T>& result, int dim, const char* comment) {
  std::cout << "\n" << comment;
  for (int i = 0; i < result.size(); ++i) {
    if (i % (dim * dim)  == 0) {
      std::cout << "\nImage: " << i / (dim*dim) << std::endl;
    }
    else if (i % dim == 0) {
      std::cout << std::endl;
    }
    std::cout << std::setw(6) << result[i];
  }
  std::cout << std::endl;
}

int main(int argc, char** argv) {

  const char filename[] = "bfly/models/cc_wide.tflite";
  TfLiteModel *model = TfLiteModelCreateFromFile(filename);
  if (model == nullptr) {
    std::cerr << "Failed to create model from file";
  }

  TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreterOptionsSetNumThreads(options, 2);
  TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);

  // Allocate tensors.
  TfLiteInterpreterAllocateTensors(interpreter);

  std::vector<float> in4{
      0.0, 0.1, 0.2, 0.3,
      1.0, 1.1, 1.2, 1.3,
      1.1, 1.1, 2.2, 2.3,
      3.0, 3.1, 2.2, 3.3,
  };

  dump(in4, 4, "4X4 input");

  TfLiteTensor *input_tensor =
      TfLiteInterpreterGetInputTensor(interpreter, 0);

//  std::cout << "## TfLiteTensorByteSize(input_tensor) = " << TfLiteTensorByteSize(input_tensor) <<std::endl;
//  std::cout << "## TfLiteTensorNumDims(input_tensor) = " << TfLiteTensorNumDims(input_tensor) <<std::endl;

  TfLiteTensorCopyFromBuffer(input_tensor, (void *) in4.data(), in4.size() * sizeof(float));

  TfLiteInterpreterInvoke(interpreter);

  std::vector<int32_t> out4(16);
  const TfLiteTensor *output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
  TfLiteTensorCopyToBuffer(output_tensor, out4.data(), out4.size() * sizeof(int32_t));


//  std::cout << "out4.size() " << out4.size() << " sizeof(int32_t) = " << sizeof(int32_t) << std::endl;
//  std::cout << " TfLiteTensorByteSize(output_tensor) " <<  TfLiteTensorByteSize(output_tensor) << std::endl;

  dump(out4, 4, "4X4 output");

  // Now resize and reallocate tensors

  int input_dims[3] = {2, 3, 3};

  TfLiteInterpreterResizeInputTensor(interpreter, 0, input_dims, 3);

  TfLiteInterpreterAllocateTensors(interpreter);

  std::vector<float> in3{
      1.1, 0.1, 1.1,
      1.1, 1.1, 1.1,
      2.0, 1.1, 2.2,

      6.0, 7.1, 6.0,
      7.1, 7.1, 7.1,
      6.0, 7.1, 6.0,
  };

  dump(in3, 3, "2 3X3 inputs");

  input_tensor =
      TfLiteInterpreterGetInputTensor(interpreter, 0);

  std::vector<int32_t> out3(18);
  output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
  TfLiteTensorCopyFromBuffer(input_tensor, (void *) in3.data(), in3.size() * sizeof(float));

  TfLiteInterpreterInvoke(interpreter);

  TfLiteTensorCopyToBuffer(output_tensor, out3.data(), out3.size() * sizeof(float));

  dump(out3, 3, "2 3X3 outputs");
}
