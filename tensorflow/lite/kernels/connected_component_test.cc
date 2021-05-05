// Tests for connected components port from tf addons.

#include <functional>
#include <memory>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_CONNECTED_COMPONENTS();


namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class ConnectedComponentOpModel : public SingleOpModel  {
 public:
  ConnectedComponentOpModel(const TensorData& input,
                            const TensorData& output) {
    input_ = AddInput(input);
    output_ = AddOutput(output);

    std::vector<uint8_t> custom_option;
    SetCustomOp("ConnectedComponent", custom_option, Register_CONNECTED_COMPONENTS);
    BuildInterpreter({GetShape(input_)});
  }

  void SetInput(const std::vector<float>& data) {
    PopulateTensor(input_, data);
  }

  std::vector<int64_t> GetOutput() { return ExtractVector<int64_t>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int output_;
};

TEST(ConnectedComponentTest, BasicTest) {
  ConnectedComponentOpModel model(
    /*input=*/{TensorType_FLOAT32, {1, 4, 4}},
    /*output=*/{TensorType_INT64, {1, 4, 4}}
  );
  model.SetInput( {
    0.0, 0.1, 0.2, 0.3,
    1.0, 1.1, 1.2, 1.3,
    1.1, 1.1, 2.2, 2.3,
    3.0, 3.1, 2.2, 3.3
  });
  model.Invoke();
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 4, 4}));
  auto output = model.GetOutput();
  // 1.1s
  EXPECT_EQ(output[5], output[8] );
  EXPECT_EQ(output[5], output[9] );
  // 2.2s
  EXPECT_EQ(output[10], output[14] );
  // Set of unique elements should be 16 - 2 - 1
  EXPECT_EQ(std::set<int64_t>(output.begin(), output.end()).size(), 16-2-1);
}

TEST(ConnectedComponentTest, TwoImageTest) {
ConnectedComponentOpModel model(
    /*input=*/{TensorType_FLOAT32, {2, 3, 3}},
    /*output=*/{TensorType_INT64, {2, 3, 3}}
);
model.SetInput( {
1.1, 0.1, 1.1,
1.1, 1.1, 1.1,
2.0, 1.1, 2.2,

6.0, 7.1, 6.0,
7.1, 7.1, 7.1,
6.0, 7.1, 6.0,
});
model.Invoke();
EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3}));
auto output = model.GetOutput();
// 1.1s in first matrix
EXPECT_EQ(output[0], output[2] );
EXPECT_EQ(output[0], output[3] );
EXPECT_EQ(output[0], output[4] );
EXPECT_EQ(output[0], output[5] );
EXPECT_EQ(output[0], output[7] );
// 7.1s in second matrix
EXPECT_EQ(output[10], output[12] );
EXPECT_EQ(output[10], output[13] );
EXPECT_EQ(output[10], output[14] );
EXPECT_EQ(output[10], output[16] );
// Set of unique elements should be 16 - 2 - 1
EXPECT_EQ(std::set<int64_t>(output.begin(), output.end()).size(), 18 - 5 - 4);
}
}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite