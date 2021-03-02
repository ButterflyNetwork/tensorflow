/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace addons {

namespace {

static const char ImageConnectedComponentsDoc[] = R"doc(
Find the connected components of image(s).
For each image (along the 0th axis), all connected components of adjacent pixels
with the same non-zero value are detected and given unique ids.
The returned `components` tensor has 0s for the zero pixels of `images`, and
arbitrary nonzero ids for the connected components of nonzero values. Ids are
unique across all of the images, and are in row-major order by the first pixel
in the component.
Uses union-find with union by rank but not path compression, giving a runtime of
`O(n log n)`. See:
    https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Time_Complexity
image: Image(s) with shape (N, H, W).
components: Component ids for each pixel in "image". Same shape as "image". Zero
    pixels all have an output of 0, and all components of adjacent pixels with
    the same value are given consecutive ids, starting from 1.
)doc";

}  // namespace


REGISTER_OP("Addons>ImageConnectedComponents")
    .Input("image: dtype")
    .Output("components: int64")
    .Attr(
        "dtype: {int64, int32, uint16, int16, uint8, int8, half, float, "
        "double, bool, string}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(ImageConnectedComponentsDoc);

}  // end namespace addons
}  // namespace tensorflow