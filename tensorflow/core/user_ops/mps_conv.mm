#define EIGEN_USE_THREADS

//#define TENSORFLOW_MPS_PRINT_DEBUG
//#define TENSORFLOW_MPS_PRINT_PERF

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/user_ops/perf.h"
#include "tensorflow/core/kernels/cast_op_impl.h"

#if TARGET_OS_IPHONE
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

#if TARGET_IPHONE_SIMULATOR == 1
#error "This operation is not supported on an iOS simulator, read tensorflow/contrib/ios_examples/mps/README.md for more details."
#endif

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;

// This is a Metal constant.
constexpr size_t max_channels_per_slice = 4;

// This is a Metal constant, after this many channels, slice overflow becomes active.
constexpr size_t depth_slice_overflow = 2;

namespace tensorflow {
namespace mps {

// Allocate a tensor that can hold mps scrambled data.
template <typename Context>
const Status alloc_scrambled(
   Context* context,
   Tensor* tensor,
   const size_t batches,
   const size_t input_height,
   const size_t input_width,
   const size_t input_slices,
   const size_t input_channels_per_slice,
   const size_t output_height,
   const size_t output_width,
   const size_t output_slices,
   const size_t output_channels_per_slice
)
{
    // MPS only supports half.
    TensorShape input_shape = ShapeFromFormat(
        FORMAT_NHWC,
        input_slices,
        input_height,
        input_width,
        input_channels_per_slice
    );
    TensorShape output_shape = ShapeFromFormat(
        FORMAT_NHWC,
        output_slices,
        output_height,
        output_width,
        output_channels_per_slice
    );

    TensorShape alloc_shape;

    if (output_shape.num_elements() > input_shape.num_elements())
    {
        alloc_shape = output_shape;
    } else {
        alloc_shape = input_shape;
    }
    return context->allocate_temp(DT_HALF, alloc_shape, tensor);
}

namespace detail {

// Helper function that computes the scrambled address.
inline const size_t compute_scrambled_address(
    size_t channel_idx, size_t channels_per_slice,
    size_t batch_idx, size_t slices_per_batch,
    size_t height_idx, size_t width_idx,
    size_t height, size_t width)
{
    const size_t out_slice = channel_idx / channels_per_slice;
    const size_t out_channel = channel_idx % channels_per_slice;
    return batch_idx * slices_per_batch * height * width * channels_per_slice +
        out_slice * height * width * channels_per_slice +
        height_idx * width * channels_per_slice +
        width_idx * channels_per_slice +
        out_channel;
}

} // namespace detail

// TODO: refactor using template:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/conv_2d.h#L139

/// MPS has a very strange data format: [slice, row, column, channel]
/// but channel must be <= 4 and if it is not, channels overflow into the
/// slice dimension.
template <typename InputTensorType>
void scramble(const TensorFormat format,
    const Tensor& input, Tensor& output,
    const size_t batches, const size_t height, const size_t width, const size_t channels,
    const size_t channels_per_slice)
{
    // Round to multiples of 4.
    const size_t slices_per_batch = (channels + channels_per_slice - 1) / channels_per_slice;

    auto inputPtr= input.flat<InputTensorType>().data();
    auto outputPtr = output.flat<Eigen::half>().data();

    if (format == FORMAT_NCHW)
    {
        // batches channel height width
        // ->
        // batches * slices height width channel
        for (size_t b = 0; b < batches; b++)
        {
            for (size_t c = 0; c < channels; c++)
            {
                for (size_t h = 0; h < height; h++)
                {
                    for (size_t w = 0; w < width; w++)
                    {
                        auto addr = detail::compute_scrambled_address(
                            c, channels_per_slice, b, slices_per_batch, h, w,
                            height, width
                        );
                        outputPtr[addr] =
                            static_cast<Eigen::half>(*inputPtr);
                        inputPtr++;
                    }
                }
            }
        }
    }
    else if (format == FORMAT_NHWC)
    {
        // batches height width channel
        // ->
        // batches * slice height width channel
        for (size_t b = 0; b < batches; b++)
        {
            for (size_t h = 0; h < height; h++)
            {
                for (size_t w = 0; w < width; w++)
                {
                    for (size_t c = 0; c < channels; c++)
                    {
                        auto addr = detail::compute_scrambled_address(
                             c, channels_per_slice, b, slices_per_batch, h, w,
                             height, width
                        );
                        outputPtr[addr] =
                            static_cast<Eigen::half>(*inputPtr);
                        inputPtr++;
                    }
                }
            }
        }
    }
    return;
}

/// MPS has a very strange data format: [slice, row, column, channel]
/// but channel must be <= 4 and if it is not, channels overflow into the
/// slice dimension.
template <typename OutputTensorType>
void unscramble(const TensorFormat format,
    const Tensor& input, Tensor& output,
    const size_t batches, const size_t height, const size_t width, const size_t channels,
    const size_t channels_per_slice)
{
    // Rount toh multiples of 4.
    const size_t slices_per_batch = (channels + channels_per_slice - 1) / channels_per_slice;


    // MPS only supports half.
    auto inputPtr= input.flat<Eigen::half>().data();
    auto outputPtr = output.flat<OutputTensorType>().data();

    if (format == FORMAT_NCHW)
    {
        // batches * slices height width channel
        // ->
        // batches channel height width
        for (size_t b = 0; b < batches; b++)
        {
            for (size_t c = 0; c < channels; c++)
            {
                for (size_t h = 0; h < height; h++)
                {
                    for (size_t w = 0; w < width; w++)
                    {
                        auto addr = detail::compute_scrambled_address(
                             c, channels_per_slice, b, slices_per_batch, h, w,
                             height, width
                        );
                        *outputPtr =
                            static_cast<OutputTensorType>(inputPtr[addr]);
                        outputPtr++;
                    }
                }
            }
        }
    }
    else if (format == FORMAT_NHWC)
    {
        // batches * slice height width channel
        // ->
        // batches height width channel
        for (size_t b = 0; b < batches; b++)
        {
            for (size_t h = 0; h < height; h++)
            {
                for (size_t w = 0; w < width; w++)
                {
                    for (size_t c = 0; c < channels; c++)
                    {
                        auto addr = detail::compute_scrambled_address(
                             c, channels_per_slice, b, slices_per_batch, h, w,
                             height, width
                        );
                        *outputPtr =
                            static_cast<OutputTensorType>(inputPtr[addr]);
                        outputPtr++;
                    }
                }
            }
        }
    }
    return;
}

} // namespace mps

namespace functor {

/// MPSCNNConvolution wants kernels ordered:
/// [output​Channels][kernel​Height][kernel​Width][input​Channels/groups]
template <typename Device, typename T, typename IndexType, int NDIMS>
struct MPSTransformFilter {
    void operator()(const Device& d,
                    typename TTypes<T, NDIMS, IndexType>::ConstTensor in,
                    typename TTypes<T, NDIMS, IndexType>::Tensor out) {
        out.device(d) = in.shuffle(Eigen::DSizes<IndexType, 4>(3, 0, 1, 2));
    }
};
} // namespace functor
} // namespace tensorflow

// Inspired by: https://github.com/tensorflow/tensorflow/blob/8746f8ac9e9ef652611180e0bf64466af2707b20/tensorflow/core/ops/nn_ops.cc#L503-L553
REGISTER_OP("Conv2DMPS")
    .Input("input: T")
    .Input("filter: T")
    .Input("bias: T")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(shape_inference::Conv2DShape);

// Inspired by: https://github.com/tensorflow/tensorflow/blob/8746f8ac9e9ef652611180e0bf64466af2707b20/tensorflow/core/kernels/conv_ops.cc#L243-L391
// TODO: Fix usage of float rather than T
// Would be cool to handle half and ints
template <typename Device, typename T>
class Conv2DMPSOp : public OpKernel {
public:
    explicit Conv2DMPSOp(OpKernelConstruction* context) : OpKernel(context) {
        const DataType dt = DataTypeToEnum<T>::v();
        OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt}, {dt}));

        OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
        string data_format;
        OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
        OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                    errors::InvalidArgument("Invalid data format"));
        OP_REQUIRES(context, strides_.size() == 4,
                    errors::InvalidArgument("Sliding window strides field must "
                                            "specify 4 dimensions"));
        const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
        const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
        OP_REQUIRES(
                    context, stride_n == 1 && stride_c == 1,
                    errors::InvalidArgument("Current implementation does not yet support "
                                            "strides in the batch and depth dimensions."));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
#if TARGET_OS_IPHONE
        device_ = MTLCreateSystemDefaultDevice();
        OP_REQUIRES(
                    context, device_ != nil,
                    errors::InvalidArgument("Could not create a Metal default device."));
        queue_ = [device_ newCommandQueue];
        OP_REQUIRES(
                    context, queue_ != nil,
                    errors::InvalidArgument("Could not create a Metal queue."));
#endif
    }

    void Compute(OpKernelContext* context) override {
        PerformanceLogger pl(true);
        pl.start("compute");
        // Input tensor is of the following dimensions:
        // [ batch, in_rows, in_cols, input_depth ]
        const Tensor& input = context->input(0);

        // Input filter is of the following dimensions:
        // [ filter_rows, filter_cols, input_depth, output_depth]
        const Tensor& filter = context->input(1);


        // For 2D convolution, there should be 4 dimensions.
        OP_REQUIRES(context, input.dims() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            input.shape().DebugString()));
        OP_REQUIRES(context, filter.dims() == 4,
                    errors::InvalidArgument("filter must be 4-dimensional: ",
                                            filter.shape().DebugString()));

        for (int i = 0; i < 3; i++) {
            OP_REQUIRES(context, FastBoundsCheck(filter.dim_size(i),
                                                 std::numeric_limits<int>::max()),
                        errors::InvalidArgument("filter too large"));
        }

        // The last dimension for input is input_depth. It must be the same as the
        // filter's input_depth.
        const int64 input_depth = GetTensorDim(input, data_format_, 'C');
        OP_REQUIRES(
                    context, input_depth == filter.dim_size(2),
                    errors::InvalidArgument("input and filter must have the same depth: ",
                                            input_depth, " vs ", filter.dim_size(2)));

        // The last dimension for filter is output_depth.
        const int output_depth = static_cast<int>(filter.dim_size(3));

        // The second dimension for input is rows/height.
        // The first dimension for filter is rows/height.
        const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
        OP_REQUIRES(context, FastBoundsCheck(input_rows_raw,
                                             std::numeric_limits<int>::max()),
                    errors::InvalidArgument("Input rows too large"));
        const int input_rows = static_cast<int>(input_rows_raw);
        const int filter_rows = static_cast<int>(filter.dim_size(0));

        // The third dimension for input is columns/width.
        // The second dimension for filter is columns/width.
        const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
        OP_REQUIRES(context, FastBoundsCheck(input_cols_raw,
                                             std::numeric_limits<int>::max()),
                    errors::InvalidArgument("Input cols too large"));
        const int input_cols = static_cast<int>(input_cols_raw);
        const int filter_cols = static_cast<int>(filter.dim_size(1));

        // The first dimension for input is batch.
        const int64 batch_raw = GetTensorDim(input, data_format_, 'N');
        OP_REQUIRES(context,
                    FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                    errors::InvalidArgument("batch is too large"));
        const int batch = static_cast<int>(batch_raw);

        // For now we take the stride from the second and third dimensions only (we
        // do not support striding on the batch or depth dimension).
        const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
        const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

        int64 output_rows = 0, output_cols = 0, pad_rows = 0, pad_cols = 0;
        OP_REQUIRES_OK(context,
                       GetWindowedOutputSize(input_rows, filter_rows, stride_rows,
                                             padding_, &output_rows, &pad_rows));
        OP_REQUIRES_OK(context,
                       GetWindowedOutputSize(input_cols, filter_cols, stride_cols,
                                             padding_, &output_cols, &pad_cols));

        const Tensor& bias = context->input(2);

        OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
                    errors::InvalidArgument("Biases must be 1D: ",
                                            bias.shape().DebugString()));

        bool isBias = bias.shape().dim_size(0) > 0;
        OP_REQUIRES(
            context,
            !isBias || bias.shape().dim_size(0) == output_depth,
            errors::InvalidArgument(
                "Must provide as many biases as the last dimension "
                "of the input tensor: ",
                bias.shape().DebugString(), " vs. ", input.shape().DebugString()));

        // This should have this format, the actual shape is fixed to match format
        TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, output_rows, output_cols, output_depth);

        // Output tensor is of the following dimensions:
        // [ in_batch, output_depth, output_rows, output_cols ]
        Tensor* output = nullptr;

        // Allocate output, unpopulated.
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

        Tensor output_fp16(DT_HALF, out_shape);
        GetCpuCastFromFloat(DT_HALF)(context, *output, &output_fp16);

#ifdef TENSORFLOW_MPS_PRINT_DEBUG
        // This was/ should be VLOG(2), but I was getting some weird link error
        std::cout << "MPSConvOp"
                  << ": batch = " << batch
                  << ", input_depth = " << input_depth
                  << ", input_cols = " << input_cols
                  << ", filter_cols = " << filter_cols
                  << ", input_rows = " << input_rows
                  << ", filter_rows = " << filter_rows
                  << ", stride_rows = " << stride_rows
                  << ", stride_cols = " << stride_cols
                  << ", output_depth = " << output_depth
                  << ", output_cols = " << output_cols
                  << ", output_rows = " << output_rows
                  << std::endl;
#endif

        // If there is nothing to compute, return.
        if (out_shape.num_elements() == 0) {
            return;
        }

        // This only works on an iPhone.
#if TARGET_OS_IPHONE == 1
        id<MTLCommandBuffer> buffer = [queue_ commandBuffer];

        // To avoid allocation (reuse excisting CPU memory):
        // -------------------------------------------------
        // make allocator page aligned so memory can be shared between GPU and CPU
        // create a metal buffer using the alread allocated tensor memory
        // create a texture from the metal buffer
        // create a MPSImage from the texture
        // docs:
        // page align memory: http://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html
        // buffer from existing memory: https://developer.apple.com/reference/metal/mtldevice/1433382-makebuffer
        // textue from buffer: https://developer.apple.com/reference/metal/mtlbuffer/1613852-maketexture
        // image from texture: https://developer.apple.com/reference/metalperformanceshaders/mpsimage/2097547-initwithtexture?language=objc



        // Transform filter.
        Tensor transformed_filter;
        {
            auto measure_copy_to_device = pl.measureScope("mps_conv2d_transform_filter");
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                       TensorShape({output_depth, filter_rows, filter_cols, input_depth}),
                                                       &transformed_filter));
            functor::MPSTransformFilter<Device, T, int, 4>()(context->eigen_device<Device>(),
                                                         To32Bit(filter.tensor<T, 4>()),
                                                         To32Bit(transformed_filter.tensor<T, 4>()));
        }

        // Set up convolution.
        // -------------------
        pl.start("setup");
        MPSCNNConvolutionDescriptor* conv1descriptor =
        [
            MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:filter_cols
            kernelHeight:filter_rows
            inputFeatureChannels:input_depth
            outputFeatureChannels:output_depth
            neuronFilter:nil
        ];

        conv1descriptor.strideInPixelsX = stride_rows;
        conv1descriptor.strideInPixelsY = stride_cols;

        MPSCNNConvolution* conv = [[MPSCNNConvolution alloc]
                    initWithDevice:device_
             convolutionDescriptor:conv1descriptor
                     kernelWeights:transformed_filter.flat<T>().data()
                         biasTerms:isBias ? bias.flat<T>().data() : nil
                             flags:MPSCNNConvolutionFlagsNone
        ];

        conv.edgeMode = MPSImageEdgeModeZero;

        MPSOffset offset;
        offset.x =  static_cast<size_t>(filter_cols/2);
        offset.y = static_cast<size_t>(filter_rows/2);
        offset.z = 0;
        conv.offset = offset;


        // Allocate input and output MPSImage.
        // -----------------------------------

        MPSImageDescriptor* inputDescriptor = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                       width:input_cols
                                      height:input_rows
                             featureChannels:input_depth
                              numberOfImages:batch
                                       usage:MTLTextureUsageShaderRead
        ];


        MPSImage* inputImage = [[MPSImage alloc] initWithDevice:device_ imageDescriptor:inputDescriptor];


        MPSImageDescriptor* outputDescriptor = [MPSImageDescriptor
                                                imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16
                                                width:output_cols
                                                height:output_rows
                                                featureChannels:output_depth
                                                numberOfImages:batch
                                                usage:MTLTextureUsageShaderWrite];

        MPSImage* outputImage = [[MPSImage alloc] initWithDevice:device_ imageDescriptor:outputDescriptor];
        pl.stop("setup");


        // Prepare input data and copy to device.
        // --------------------------------------

        /*
         If feature​Channels<=4 and number​Of​Images=1 (i.e. only one slice is needed to represent the image),
         the underlying metal texture type is chosen to be type2D rather than type2DArray. MTLTexture​Type2D 
         has a single slice.

         The number of feature channels does not change but the number of underlying array slices is (fchannels+3)/4.

         Convolution descriptor and image descriptors are not affected by this but the underlying textures and the
         data buffers that feed the textures neet do take this into consideration.
         
         If the depth is > 2 MPS overflows into RGBA textures and slices must be used.
        */

        size_t input_channels_per_slice = input_depth;
        size_t input_slices = batch;
        if (input_depth > depth_slice_overflow)
        {
            input_channels_per_slice = max_channels_per_slice;
            input_slices = batch * ((input_depth + max_channels_per_slice-1) / max_channels_per_slice);
        }

        size_t output_channels_per_slice = output_depth;
        size_t output_slices = batch;
        if (output_depth > depth_slice_overflow)
        {
            output_channels_per_slice = max_channels_per_slice;
            output_slices = batch * ((output_depth + max_channels_per_slice-1) / max_channels_per_slice);
        }

        // Allocate a temporary tensor.
        Tensor tmp_tensor;
        OP_REQUIRES_OK(context,
            mps::alloc_scrambled(context, &tmp_tensor,
                batch, input_rows, input_cols, input_slices, input_channels_per_slice,
                output_rows, output_cols, output_slices, output_channels_per_slice
            )
        );

        // Scramble the input so that MPS can process it.
        {
            auto measure_copy_to_device = pl.measureScope("mps_conv2d_scramble");
            mps::scramble<T>(data_format_,
                input, tmp_tensor, batch, input_rows, input_cols, input_depth, input_channels_per_slice);
        }

#ifdef TENSORFLOW_MPS_PRINT_DEBUG
        {
            std::cout << "Input -----------------------------------------------";
            std::cout << "batch " << batch <<
                " input_rows " << input_rows <<
                " input_cols " << input_cols <<
                " input_slices " << input_slices <<
                " input_channels_per_slice " << input_channels_per_slice << std::endl;

            auto t= tmp_tensor.flat<Eigen::half>().data();
            for (int i = 0; i < batch*input_rows*input_cols*input_slices*input_channels_per_slice; i++)
            {
                if (i % (4 * input_cols) == 0)
                    std::cout << std::endl;
                std::cout << static_cast<float>(t[i]) << " ";

            }
            std::cout <<
                std::endl <<
                "------------------------------------------------------" <<
                std::endl;
        }
#endif

        // Populate input.
        {
            auto measure_copy_to_device = pl.measureScope("mps_conv2d_copy_to_device");
            long values_per_row = input_cols * input_channels_per_slice;
            long bytes_per_row = sizeof(Eigen::half) * values_per_row;
#ifdef TENSORFLOW_MPS_PRINT_DEBUG
            std::cout << "values_per_row " << values_per_row << std::endl;
#endif
            for (int slice = 0; slice < input_slices; slice++)
            {
                [inputImage.texture
                    replaceRegion:MTLRegionMake3D(0, 0, 0, input_cols, input_rows, 1)
                      mipmapLevel:0
                            slice:slice
                        withBytes:tmp_tensor.flat<Eigen::half>().data() +
                            slice * values_per_row * input_cols
                      bytesPerRow:bytes_per_row
                    bytesPerImage:0
                ];
            }
        }


        {
            auto measure_compute = pl.measureScope("mps_conv2d__compute");
            [conv encodeToCommandBuffer:buffer sourceImage:inputImage destinationImage:outputImage];
            [buffer commit];
            [buffer waitUntilCompleted];
        }
        
        // Get out result.
        {
            auto measure_copy_to_device = pl.measureScope("mps_conv2d__copy_from_device");
            long values_per_row = output_cols * output_channels_per_slice;
            long bytes_per_row = sizeof(Eigen::half) * values_per_row;
#ifdef TENSORFLOW_MPS_PRINT_DEBUG
            std::cout << "output_slices " << output_slices <<
                " slize size bytes " << values_per_row * output_rows <<
                " values_per_row" << values_per_row << std::endl;
#endif
            for (int slice = 0; slice < output_slices; slice++)
            {
                [outputImage.texture
                         getBytes:tmp_tensor.flat<Eigen::half>().data() +
                            slice * values_per_row * output_rows
                      bytesPerRow:bytes_per_row
                    bytesPerImage:0
                       fromRegion:MTLRegionMake3D(0, 0, 0, output_cols, output_rows, 1)
                      mipmapLevel:0
                            slice:slice
                ];
            }
        }

        // Unscramble output.
        {
            auto measure_copy_to_device = pl.measureScope("mps_conv2d__unscramble");

            mps::unscramble<T>(data_format_, tmp_tensor, *output, batch, output_rows, output_cols, output_depth,
                          output_channels_per_slice);
        }

#ifdef TENSORFLOW_MPS_PRINT_DEBUG
        {
            std::cout << "Output -----------------------------------------------";
            auto t= output->flat<T>().data();
            //auto t= tmp_tensor.flat<Eigen::half>().data();

            for (int i = 0; i < batch*output_cols*output_rows*output_slices*output_channels_per_slice; i++)
            {
                if (i % (4 * output_cols) == 0)
                    std::cout << std::endl;
                std::cout << t[i] << " ";

            }
            std::cout << std::endl << "------------------------------------------------------" << std::endl;
        }
#endif
        pl.stop("compute");
#ifdef TENSORFLOW_MPS_PRINT_PERF
        std::cout << name() << "\n" << pl.getSignalsAsString() << std::endl;
#endif
#endif
}

private:
    std::vector<int32> strides_;
    Padding padding_;
    TensorFormat data_format_;
#if TARGET_OS_IPHONE == 1
    id<MTLCommandQueue> queue_;
    id<MTLDevice> device_;
#endif

    TF_DISALLOW_COPY_AND_ASSIGN(Conv2DMPSOp);

};

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
  Name("Conv2DMPS").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
  Conv2DMPSOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU);
