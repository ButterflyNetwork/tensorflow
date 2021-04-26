// This is a port of the connected_components module from
// tensorflow_addons. Input and output shapes are the same.
// We expect a shape of (N, H, W). The code supports both
// integer and float types. Output type is int64.
//
#include <iostream>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_threadpool.h"
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


template <typename T>
bool is_nonzero(T value) {
  return value != T(0);
}


// Processes each pixel of an image for union-find, in parallel blocks. This is
// loosely based on the algorithm in "GPU Computing Gems" by Ondrej Stava and
// Bedrich Benes, available here:
// http://hpcg.purdue.edu/bbenes/papers/Stava2011CCL.pdf
// The bulk of the process uses blocks of each image, which have each been
// processed separately. As long as there are multiple blocks in the image, we
// double the height and width of the blocks, creating new blocks which each
// consist of 2x2 previous sub-blocks. On each new block, we process adjacent
// pixels from the previous sub-blocks serially. However, the new blocks are not
// connected, so we can process each block in parallel.
// The GPU algorithm first processes blocks of a fixed size in GPU shared
// memory, with one image block per CUDA thread block. On the CPU, we just start
// with a block size of a single pixel, and borrow the rest of the algorithm
// unchanged.
template <typename T>
class BlockedImageUnionFindFunctor {
 public:
  using OutputType = int64_t;

  BlockedImageUnionFindFunctor(
      const T* images, const int64_t num_rows, const int64_t num_cols,
      OutputType* forest, OutputType* rank)
      : images_(images),
        num_rows_(num_rows),
        num_cols_(num_cols),
        block_height_(1),
        block_width_(1),
        forest_(forest),
        rank_(rank) {}

  // Returns the root of the tree that the pixel at the given index belongs to.
  OutputType
  find(OutputType index) const {
    while (forest_[index] != index) {
      index = forest_[index];
    }
    return index;
  }

  // Returns the number of blocks along the y axis.
  int64_t num_blocks_vertically() const {
    return (num_rows_ + block_height_ - 1) / block_height_;
  }

  // Returns the number of blocks along the x axis.
  int64_t num_blocks_horizontally() const {
    return (num_cols_ + block_width_ - 1) / block_width_;
  }

  // Returns the total number of blocks in each image.
  int64_t num_blocks() const {
    return num_blocks_vertically() * num_blocks_horizontally();
  }

  int64_t block_height() const {
    return block_height_;
  }

  int64_t block_width() const {
    return block_width_;
  }

  // Returns whether we may merge again (the image contains more than one
  // block).
  bool can_merge() const {
    return block_height_ < num_rows_ || block_width_ < num_cols_;
  }

  // Doubles the block size. After this method, you must call
  // `merge_internal_block_edges` for each image and each *new* block's xy
  // coordinates (typically in parallel).
  void merge_blocks() {
    block_height_ *= 2;
    block_width_ *= 2;
  }

  // Processes pairs of pixels within the block which were adjacent in the four
  // sub-blocks. This must be done at each stage so that the connected
  // components in each block are joined correctly.
  void merge_internal_block_edges(
      int64_t image_index, int64_t block_vertical_index,
      int64_t block_horizontal_index) const {
    int64_t block_start_y = block_vertical_index * block_height_;
    int64_t block_start_x = block_horizontal_index * block_width_;
    // Merge the 4 sub-blocks horizontally (fixing the vertical seam).
    int64_t block_center_x = block_start_x + block_width_ / 2 - 1;
    if (0 <= block_center_x && block_center_x + 1 < num_cols_) {
      int64_t merge_blocks_limit_y =
          std::min(num_rows_, block_start_y + block_height_);
      for (int64_t y = block_start_y; y < merge_blocks_limit_y; y++) {
        union_right(image_index, y, block_center_x);
      }
    }
    // Merge the 4 sub-blocks vertically (fixing the horizontal seam).
    int64_t block_center_y = block_start_y + block_height_ / 2 - 1;
    if (0 <= block_center_y && block_center_y + 1 < num_rows_) {
      int64_t merge_blocks_limit_x =
          std::min(num_cols_, block_start_x + block_width_);
      for (int64_t x = block_start_x; x < merge_blocks_limit_x; x++) {
        union_down(image_index, block_center_y, x);
      }
    }
  }

 private:
  // The input image(s).
  const T* const images_;
  const int64_t num_rows_;
  const int64_t num_cols_;
  // Current height of each sub-block of the image.
  int64_t block_height_;
  // Current width of each sub-block of the image.
  int64_t block_width_;
  // Union-find forest. This has the same size as `images_`, and each entry
  // holds the index of its parent in `images_` (roots hold their own index).
  // Cycles should not occur.
  OutputType* const forest_;
  // Union-find rank of each pixel.
  OutputType* const rank_;

  // Unions the pixel with the pixel below it if applicable (both pixels are
  // true, and the pixel is not in the last row).
  void union_down(OutputType batch, OutputType row, OutputType col) const {
    T pixel = read_pixel(batch, row, col);
    if (is_nonzero<T>(pixel)) {
      const int64_t index_a = col + num_cols_ * (row + num_rows_ * batch);
      if (row + 1 < num_rows_ && read_pixel(batch, row + 1, col) == pixel) {
        const int64_t index_b = col + num_cols_ * (row + 1 + num_rows_ * batch);
        do_union(index_a, index_b);
      }
    }
  }

  // Unions the pixel with the pixel to the right of it if applicable.
  void union_right(OutputType batch, OutputType row, OutputType col) const {
    T pixel = read_pixel(batch, row, col);
    if (is_nonzero<T>(pixel)) {
      const int64_t index_a = col + num_cols_ * (row + num_rows_ * batch);
      if (col + 1 < num_cols_ && read_pixel(batch, row, col + 1) == pixel) {
        const int64_t index_b = col + 1 + num_cols_ * (row + num_rows_ * batch);
        do_union(index_a, index_b);
      }
    }
  }

  // Reads a pixel value in the images.
  T
  read_pixel(const OutputType batch, const OutputType row,
             const OutputType col) const {
    return images_[col + num_cols_ * (row + num_rows_ * batch)];
  }

  // Unions the trees that the two pixels belong to, using their index in the
  // `images_` array.
  void do_union(
      OutputType index_a, OutputType index_b) const {
    // Find the roots of index_a and index_b in the forest, and make one the
    // child of the other.
    index_a = find(index_a);
    index_b = find(index_b);
    const OutputType rank_a = rank_[index_a];
    const OutputType rank_b = rank_[index_b];
    OutputType parent, child;
    if (index_a == index_b) {
      return;
    } else if (rank_a < rank_b) {
      parent = index_a;
      child = index_b;
    } else {
      parent = index_b;
      child = index_a;
      rank_[parent]++;
    }
    forest_[child] = parent;
  }
};

// Struct to hold temporary Tensors
struct OpData {
  int scratch_tensor_index;
};


template <typename T>
struct ConnectedComponentWorkerTask : cpu_backend_threadpool::Task {
  ConnectedComponentWorkerTask(
      BlockedImageUnionFindFunctor<T>* union_find,
      int64_t start_block,
      int64_t limit_block,
      int64_t num_blocks_vertically,
      int64_t num_blocks_horizontally
  ):
    union_find_(union_find),
    start_block_(start_block),
    limit_block_{limit_block},
    num_blocks_vertically_{num_blocks_vertically},
    num_blocks_horizontally_(num_blocks_horizontally) {}

  void Run() override {
    for (int64_t i = start_block_; i < limit_block_; i++) {
      int64_t block_x = i % num_blocks_horizontally_;
      int64_t block_y =
          (i / num_blocks_horizontally_) % num_blocks_vertically_;
      int64_t image =
          i / (num_blocks_horizontally_ * num_blocks_vertically_);
      union_find_->merge_internal_block_edges(image, block_y, block_x);
    }
  }

 private:
  BlockedImageUnionFindFunctor<T>* union_find_;
  int64_t start_block_;
  int64_t limit_block_;
  int64_t num_blocks_vertically_;
  int64_t num_blocks_horizontally_;
};


}  // namespace



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

  const TfLiteTensor *images_t;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &images_t));
  TfLiteTensor *output_t;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output_t));

  TF_LITE_ENSURE_EQ(context, images_t->dims->size, 3);
  TF_LITE_ENSURE_TYPES_EQ(context, images_t->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, output_t->type, kTfLiteInt64);

  // allocate temporary tensors
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(kTempTensorCount);

  // forest_t temporary
  node->temporaries->data[0] = op_data->scratch_tensor_index;
  TfLiteTensor* forest_t;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/0, &forest_t));
  forest_t->type = kTfLiteInt64;
  forest_t->allocation_type = kTfLiteArenaRw;
  if (!TfLiteIntArrayEqual(forest_t->dims, images_t->dims)) {
    TfLiteIntArray* input_size = TfLiteIntArrayCopy(images_t->dims);
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, forest_t, input_size));
  }

  // rank_t temporary
  node->temporaries->data[1] = op_data->scratch_tensor_index+1;
  TfLiteTensor* rank_t;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/1, &rank_t));
  rank_t->type = kTfLiteInt64;
  rank_t->allocation_type = kTfLiteArenaRw;
  if (!TfLiteIntArrayEqual(rank_t->dims, images_t->dims)) {
    TfLiteIntArray* input_size = TfLiteIntArrayCopy(images_t->dims);
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, rank_t, input_size));
  }

  // resize output as necessary
  if(!TfLiteIntArrayEqual(output_t->dims, images_t->dims)) {
    TfLiteIntArray *input_size = TfLiteIntArrayCopy(images_t->dims);
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_t, input_size));
  }
  return kTfLiteOk;
}


TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  TfLiteTensor *output_t;
  TF_LITE_ENSURE_OK(context,GetOutputSafe(context, node, kOutputTensor, &output_t));
  const TfLiteTensor *images_t;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &images_t));
  const int flat_size = GetTensorShape(output_t).FlatSize();
  auto *output_data = GetTensorData<int64_t>(output_t);

  TfLiteTensor* forest_t;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/0, &forest_t));
  auto *forest_data = GetTensorData<int64_t>(forest_t);

  TfLiteTensor* rank_t;
  TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/1, &rank_t));
  auto *rank_data = GetTensorData<int64_t>(rank_t);

  
  // Fill forest with values from 0 to n-1
  for (int i = 0; i < flat_size; ++i) {
    forest_data[i] = int64_t{i};
    rank_data[i] = int64_t{0};
  }

  CpuBackendContext* cpu_backend_context =
      CpuBackendContext::GetFromContext(context);
  const int thread_count = cpu_backend_context->max_num_threads();
  std::cout << "### thread count = " << thread_count << std::endl;

  auto num_images = images_t->dims->data[0];
  auto num_rows = images_t->dims->data[1];
  auto num_cols = images_t->dims->data[2];
  auto *images_data = GetTensorData<float_t>(images_t);

  BlockedImageUnionFindFunctor<float_t> union_find(
      images_data, num_rows, num_cols, forest_data, rank_data
  );

  while(union_find.can_merge()) {
    union_find.merge_blocks();
    int64_t num_blocks_vertically = union_find.num_blocks_vertically();
    int64_t num_blocks_horizontally = union_find.num_blocks_horizontally();
    std::vector<ConnectedComponentWorkerTask<float_t>> tasks;
    // tasks.reserve(thread_count);
    // pinning to one thread for now.
    tasks.reserve(1);
    // Replicating the tf-addons logic here - that is using "cost" and
    // striding is a bit awkward.  More than likely this algorithm is
    // sort of inefficient given the low number of available (cpu) threads.
    // But here's a rough go at this.
    auto total = num_images * num_blocks_vertically * num_blocks_horizontally;
    tasks.emplace_back(
        ConnectedComponentWorkerTask<float_t>(
          &union_find,
          0,
          total,
          num_blocks_vertically,
          num_blocks_horizontally
        )
    );
    cpu_backend_threadpool::Execute(tasks.size(), tasks.data(), cpu_backend_context);
  }


  // some bogus output for now.
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