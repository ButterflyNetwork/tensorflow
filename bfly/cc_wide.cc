//
// Created by Benjamin Sternlieb on 4/21/21.
//
#include <iostream>
#include <vector>

#include "tensorflow/lite/c/c_api.h"


int main(int argc, char** argv) {

  const char filename[] = "bfly/models/cc_wide.tflite";
  TfLiteModel* model = TfLiteModelCreateFromFile(filename);
  if (model == nullptr) {
    std::cerr << "Failed to create model from file";
  }

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // Allocate tensors.
  TfLiteInterpreterAllocateTensors(interpreter);

  std::vector<float> in4 {
      0.0, 0.1, 0.2, 0.3,
      1.0, 1.1, 1.2, 1.3,
      2.0, 2.1, 2.2, 2.3,
      3.0, 3.1, 3.2, 3.3,
  };


  TfLiteTensor* input_tensor =
      TfLiteInterpreterGetInputTensor(interpreter, 0);

  std::cout << "## TfLiteTensorByteSize(input_tensor) = " << TfLiteTensorByteSize(input_tensor) <<std::endl;
  std::cout << "## TfLiteTensorNumDims(input_tensor) = " << TfLiteTensorNumDims(input_tensor) <<std::endl;

  TfLiteTensorCopyFromBuffer(input_tensor, (void *)in4.data(),  in4.size() * sizeof(float));

  TfLiteInterpreterInvoke(interpreter);


  std::vector<int32_t> out4(16);

  const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
  TfLiteTensorCopyToBuffer(output_tensor, out4.data(), out4.size() * sizeof(int32_t));


  std::cout << "out4.size() " << out4.size() << " sizeof(int32_t) = " << sizeof(int32_t) << std::endl;
  std::cout << " TfLiteTensorByteSize(output_tensor) " <<  TfLiteTensorByteSize(output_tensor) << std::endl;

  std::cout << "\n 4X4 data " << std::endl;
  for (auto& x : out4) {
    std::cout << x << std::endl;

  }

  // Now resize and reallocate tensors

  int input_dims[3] = {1, 3, 3};

  std::cout << "## Just before calling resize" << std::endl;

  TfLiteInterpreterResizeInputTensor(interpreter, 0, input_dims, 3);

  std::cout << "## Just after calling resize" << std::endl;

  std::cout << "## Just before Allocate Tensors ##" << std::endl;
  TfLiteInterpreterAllocateTensors(interpreter);
  std::cout << "## Just after Allocate Tensors ##" << std::endl;

  std::vector<float> in3 {
      0.0, 0.1, 0.2,
      1.0, 1.1, 1.2,
      2.0, 2.1, 2.2,
  };

  input_tensor =
      TfLiteInterpreterGetInputTensor(interpreter, 0);

  std::vector<int32_t> out3(9);
  output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

  std::cout << "## have input tensor" << std::endl;
  std::cout << "## TfLiteTensorByteSize(input_tensor) = " << TfLiteTensorByteSize(input_tensor) <<std::endl;
  std::cout << "## TfLiteTensorNumDims(input_tensor) = " << TfLiteTensorNumDims(input_tensor) <<std::endl;


  TfLiteTensorCopyFromBuffer(input_tensor, (void *)in3.data(),  in3.size() * sizeof(float));

  std::cout << "## just before invoke" << std::endl;

  TfLiteInterpreterInvoke(interpreter);



  TfLiteTensorCopyToBuffer(output_tensor, out3.data(), out3.size() * sizeof(float));

  std::cout << "\n 3X3 data " << std::endl;
  for (auto& x : out3) {
    std::cout << x << std::endl;
  }

  std::cout << "Here!" << std::endl;
}

