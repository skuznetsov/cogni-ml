// Diagnostic-only bridge for the pinned PyTorch 2.8 MPS Conv1d arithmetic.
//
// The caller owns every MTLBuffer and the MPSCommandBuffer. A plan retains
// graph feeds/results and buffer references once, then encode only appends the
// prepared MPSGraph operation. MPSGraph may call commitAndContinue internally;
// callers must refresh their raw MTLCommandBuffer from the MPSCommandBuffer's
// rootCommandBuffer after encode and recreate any Metal encoder.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <stdint.h>
#include <limits.h>

@interface CogniWanAudioMPSConv1dPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLBuffer> input;
@property(nonatomic, strong) id<MTLBuffer> weight;
@property(nonatomic, strong) id<MTLBuffer> bias;
@property(nonatomic, strong) id<MTLBuffer> output;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) MPSGraphTensor *inputTensor;
@property(nonatomic, strong) MPSGraphTensor *weightTensor;
@property(nonatomic, strong) MPSGraphTensor *biasTensor;
@property(nonatomic, strong) MPSGraphTensor *outputTensor;
@property(nonatomic, strong) MPSGraphTensorData *inputData;
@property(nonatomic, strong) MPSGraphTensorData *weightData;
@property(nonatomic, strong) MPSGraphTensorData *biasData;
@property(nonatomic, strong) MPSGraphTensorData *outputData;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic) NSUInteger inputFrames;
@property(nonatomic) NSUInteger inputChannels;
@property(nonatomic) NSUInteger outputChannels;
@property(nonatomic) NSUInteger kernelWidth;
@property(nonatomic) NSUInteger stride;
@property(nonatomic) NSUInteger outputFrames;
@end

@implementation CogniWanAudioMPSConv1dPlan
@end

static bool wan_audio_bytes(int32_t a, int32_t b, int32_t c, uint64_t *out) {
  if (a <= 0 || b <= 0 || c <= 0) return false;
  uint64_t count = (uint64_t)a;
  if (count > (uint64_t)INT32_MAX / (uint64_t)b) return false;
  count *= (uint64_t)b;
  if (count > (uint64_t)INT32_MAX / (uint64_t)c) return false;
  count *= (uint64_t)c;
  if (count > (uint64_t)INT32_MAX / sizeof(float)) return false;
  *out = count * sizeof(float);
  return true;
}

static bool wan_audio_disjoint(id<MTLBuffer> a, id<MTLBuffer> b) {
  if (a == b || a.gpuAddress == 0 || b.gpuAddress == 0) return false;
  const uint64_t a0 = a.gpuAddress, b0 = b.gpuAddress;
  const uint64_t a1 = a0 + a.length, b1 = b0 + b.length;
  return a1 >= a0 && b1 >= b0 && (a1 <= b0 || b1 <= a0);
}

extern "C" void *gs_wan_audio_mps_conv1d_plan_create(
    void *device_handle, void *input_handle, void *weight_handle,
    void *bias_handle, void *output_handle, int32_t input_frames,
    int32_t input_channels, int32_t output_channels, int32_t kernel_width,
    int32_t stride) {
  if (!device_handle || !input_handle || !weight_handle || !bias_handle ||
      !output_handle || input_frames <= 0 || input_channels <= 0 ||
      output_channels <= 0 || kernel_width <= 0 || stride <= 0 ||
      input_frames < kernel_width) return nullptr;

  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      const int32_t output_frames =
          (input_frames - kernel_width) / stride + 1;
      uint64_t input_bytes = 0, weight_bytes = 0, bias_bytes = 0,
               output_bytes = 0;
      if (!wan_audio_bytes(input_frames, input_channels, 1, &input_bytes) ||
          !wan_audio_bytes(output_channels, input_channels, kernel_width,
                           &weight_bytes) ||
          !wan_audio_bytes(output_channels, 1, 1, &bias_bytes) ||
          !wan_audio_bytes(output_frames, output_channels, 1, &output_bytes))
        return nullptr;

      id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
      id<MTLBuffer> input = (__bridge id<MTLBuffer>)input_handle;
      id<MTLBuffer> weight = (__bridge id<MTLBuffer>)weight_handle;
      id<MTLBuffer> bias = (__bridge id<MTLBuffer>)bias_handle;
      id<MTLBuffer> output = (__bridge id<MTLBuffer>)output_handle;
      if (input.device != device || weight.device != device ||
          bias.device != device || output.device != device ||
          input.length < input_bytes || weight.length < weight_bytes ||
          bias.length < bias_bytes || output.length < output_bytes ||
          !wan_audio_disjoint(input, weight) ||
          !wan_audio_disjoint(input, bias) ||
          !wan_audio_disjoint(input, output) ||
          !wan_audio_disjoint(weight, bias) ||
          !wan_audio_disjoint(weight, output) ||
          !wan_audio_disjoint(bias, output)) return nullptr;

      @try {
        CogniWanAudioMPSConv1dPlan *plan = [CogniWanAudioMPSConv1dPlan new];
        plan.device = device;
        plan.input = input;
        plan.weight = weight;
        plan.bias = bias;
        plan.output = output;
        plan.inputFrames = (NSUInteger)input_frames;
        plan.inputChannels = (NSUInteger)input_channels;
        plan.outputChannels = (NSUInteger)output_channels;
        plan.kernelWidth = (NSUInteger)kernel_width;
        plan.stride = (NSUInteger)stride;
        plan.outputFrames = (NSUInteger)output_frames;

        // Sequence-major caller input [T,C] is the physical storage left by
        // HF's per-frame LayerNorm. Its logical Conv1d view is [1,C,1,T].
        // Keep that transpose inside the graph: no caller or bridge MTLBuffer
        // is allocated for a layout conversion.
        plan.graph = [[MPSGraph alloc] init];
        plan.graph.options = MPSGraphOptionsNone;
        MPSShape *inputShape = @[@1, @1, @(input_frames), @(input_channels)];
        plan.inputTensor = [plan.graph placeholderWithShape:inputShape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"sequence_major_input"];
        MPSGraphTensor *inputSwapChannelsWidth = [plan.graph
            transposeTensor:plan.inputTensor dimension:1 withDimension:3
            name:@"input_swap_channels_width"];
        MPSGraphTensor *channelsFirst = [plan.graph
            transposeTensor:inputSwapChannelsWidth dimension:2 withDimension:3
            name:@"input_to_nchw"];

        // PyTorch's Conv1d weight [O,C,K] is interpreted as OIHW [O,C,1,K].
        MPSShape *weightShape = @[@(output_channels), @(input_channels), @1,
                                  @(kernel_width)];
        plan.weightTensor = [plan.graph placeholderWithShape:weightShape
                                                     dataType:MPSDataTypeFloat32
                                                         name:@"conv_weight_oihw"];
        MPSShape *biasShape = @[@1, @(output_channels), @1, @1];
        plan.biasTensor = [plan.graph placeholderWithShape:biasShape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"conv_bias_nchw"];
        // Match pytorch/pytorch v2.8.0
        // aten/src/ATen/native/mps/operations/Convolution.mm: explicit NCHW /
        // OIHW descriptor, convolution op, then a separate broadcast add.
        MPSGraphConvolution2DOpDescriptor *descriptor =
            [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:(NSUInteger)stride
                strideInY:1
                dilationRateInX:1
                dilationRateInY:1
                groups:1
                paddingLeft:0
                paddingRight:0
                paddingTop:0
                paddingBottom:0
                paddingStyle:MPSGraphPaddingStyleExplicit
                dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        if (!descriptor) return nullptr;
        MPSGraphTensor *convolved = [plan.graph
            convolution2DWithSourceTensor:channelsFirst
            weightsTensor:plan.weightTensor descriptor:descriptor name:@"conv1d_as_conv2d"];

        // PyTorch 2.8 Convolution.mm adds the broadcast bias as a distinct
        // MPSGraph operation after convolution (not as a convolution seed).
        MPSGraphTensor *biased = [plan.graph
            additionWithPrimaryTensor:convolved secondaryTensor:plan.biasTensor
            name:@"broadcast_bias_after_conv"];
        MPSGraphTensor *sequenceMajor = [plan.graph
            transposeTensor:biased dimension:1 withDimension:3
            name:@"output_to_sequence_major"];
        plan.outputTensor = [plan.graph
            reshapeTensor:sequenceMajor
            withShape:@[@(output_frames), @(output_channels)]
            name:@"sequence_major_output"];

        // Compile here so the first caller encode does not compile/specialize
        // this shape. MTLBuffer bindings are then created once with the plan
        // and point directly at caller-owned resident storage.
        MPSGraphShapedType *inputType = [[MPSGraphShapedType alloc]
            initWithShape:inputShape dataType:MPSDataTypeFloat32];
        MPSGraphShapedType *weightType = [[MPSGraphShapedType alloc]
            initWithShape:weightShape dataType:MPSDataTypeFloat32];
        MPSGraphShapedType *biasType = [[MPSGraphShapedType alloc]
            initWithShape:biasShape dataType:MPSDataTypeFloat32];
        if (!inputType || !weightType || !biasType) return nullptr;
        MPSGraphTensorShapedTypeDictionary *feedTypes = @{
          plan.inputTensor : inputType,
          plan.weightTensor : weightType,
          plan.biasTensor : biasType,
        };
        plan.executable = [plan.graph
            compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                        feeds:feedTypes
                targetTensors:@[plan.outputTensor]
             targetOperations:nil
        compilationDescriptor:nil];
        if (!plan.executable) return nullptr;
        plan.executable.options = MPSGraphOptionsNone;

        // No graph sync-to-CPU is enabled. These TensorData instances retain
        // the caller's MTLBuffers and are reused for every serial encode.
        plan.inputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input shape:inputShape dataType:MPSDataTypeFloat32];
        plan.weightData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:weight shape:weightShape dataType:MPSDataTypeFloat32];
        plan.biasData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bias shape:biasShape dataType:MPSDataTypeFloat32];
        plan.outputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:output
            shape:@[@(output_frames), @(output_channels)]
            dataType:MPSDataTypeFloat32];
        if (!plan.inputData || !plan.weightData || !plan.biasData ||
            !plan.outputData || !plan.outputTensor ||
            plan.executable.feedTensors.count != 3 ||
            plan.executable.targetTensors.count != 1 ||
            plan.executable.targetTensors.firstObject != plan.outputTensor)
          return nullptr;

        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feedData = @{
          plan.inputTensor : plan.inputData,
          plan.weightTensor : plan.weightData,
          plan.biasTensor : plan.biasData,
        };
        NSMutableArray<MPSGraphTensorData *> *orderedInputs =
            [NSMutableArray arrayWithCapacity:3];
        for (MPSGraphTensor *feedTensor in plan.executable.feedTensors) {
          MPSGraphTensorData *feedValue = feedData[feedTensor];
          if (!feedValue) return nullptr;
          [orderedInputs addObject:feedValue];
        }
        plan.inputArray = [orderedInputs copy];
        plan.outputArray = @[plan.outputData];
        plan.execution = [MPSGraphExecutableExecutionDescriptor new];
        plan.execution.waitUntilCompleted = NO;
        if (!plan.inputArray || !plan.outputArray || !plan.execution)
          return nullptr;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

// `command_handle` is a caller-owned MPSCommandBuffer wrapper. MPSGraph may
// commitAndContinue; use gs_wan_audio_mps_command_buffer_copy_root afterward
// and recreate any Metal encoder on that root before appending more work.
extern "C" int32_t gs_wan_audio_mps_conv1d_encode(
    void *plan_handle, void *command_handle) {
  if (!plan_handle || !command_handle) return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      CogniWanAudioMPSConv1dPlan *plan =
          (__bridge CogniWanAudioMPSConv1dPlan *)plan_handle;
      MPSCommandBuffer *command = (__bridge MPSCommandBuffer *)command_handle;
      id<MTLCommandBuffer> root = command.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted) return -2;
      @try {
      NSArray<MPSGraphTensorData *> *results =
          [plan.executable encodeToCommandBuffer:command
                                     inputsArray:plan.inputArray
                                    resultsArray:plan.outputArray
                             executionDescriptor:plan.execution];
      if (results.count != 1) return -4;
      } @catch (NSException *exception) {
        (void)exception;
        return -5;
      }
      // The caller deliberately adopts the possibly-rotated MPS root.
      return 0;
    }
  }
  return -6;
}

// Returns an owned opaque MTLCommandBuffer handle. The caller must release it
// with its ordinary command-buffer release function after adopting/waiting.
extern "C" void *gs_wan_audio_mps_command_buffer_copy_root(
    void *command_handle) {
  if (!command_handle) return nullptr;
  MPSCommandBuffer *command = (__bridge MPSCommandBuffer *)command_handle;
  id<MTLCommandBuffer> root = command.rootCommandBuffer;
  return root ? (__bridge_retained void *)root : nullptr;
}

extern "C" void gs_wan_audio_mps_conv1d_plan_release(void *plan_handle) {
  if (plan_handle) {
    id plan = (__bridge_transfer id)plan_handle;
    (void)plan;
  }
}

// Prepared grouped positional Conv1d for Wav2Vec2. The caller supplies the
// already-derived per-kernel weight-normalization scale, so this isolates the
// convolution/activation implementation from the native scale reduction.
// Inputs/outputs are caller-owned resident buffers and remain retained by the
// plan through terminal command completion.
@interface CogniWanAudioMPSPositionPlan : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLBuffer> input;
@property(nonatomic, strong) id<MTLBuffer> weightV;
@property(nonatomic, strong) id<MTLBuffer> scale;
@property(nonatomic, strong) id<MTLBuffer> bias;
@property(nonatomic, strong) id<MTLBuffer> rawOutput;
@property(nonatomic, strong) id<MTLBuffer> output;
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphExecutable *executable;
@property(nonatomic, strong) MPSGraphTensor *inputTensor;
@property(nonatomic, strong) MPSGraphTensor *weightTensor;
@property(nonatomic, strong) MPSGraphTensor *scaleTensor;
@property(nonatomic, strong) MPSGraphTensor *biasTensor;
@property(nonatomic, strong) MPSGraphTensor *rawTensor;
@property(nonatomic, strong) MPSGraphTensor *outputTensor;
@property(nonatomic, strong) MPSGraphTensorData *inputData;
@property(nonatomic, strong) MPSGraphTensorData *weightData;
@property(nonatomic, strong) MPSGraphTensorData *scaleData;
@property(nonatomic, strong) MPSGraphTensorData *biasData;
@property(nonatomic, strong) MPSGraphTensorData *rawOutputData;
@property(nonatomic, strong) MPSGraphTensorData *outputData;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *inputArray;
@property(nonatomic, strong) NSArray<MPSGraphTensorData *> *outputArray;
@property(nonatomic, strong) MPSGraphExecutableExecutionDescriptor *execution;
@property(nonatomic) NSUInteger frames;
@property(nonatomic) NSUInteger hidden;
@property(nonatomic) NSUInteger groups;
@property(nonatomic) NSUInteger kernelWidth;
@end

@implementation CogniWanAudioMPSPositionPlan
@end

static bool wan_audio_position_bytes(int32_t frames, int32_t hidden,
                                     int32_t groups, int32_t kernel_width,
                                     uint64_t *input_bytes,
                                     uint64_t *weight_bytes,
                                     uint64_t *scale_bytes,
                                     uint64_t *bias_bytes,
                                     uint64_t *output_bytes) {
  if (frames <= 0 || hidden <= 0 || groups <= 0 || kernel_width <= 0 ||
      hidden % groups != 0) return false;
  const int32_t group_channels = hidden / groups;
  return wan_audio_bytes(frames, hidden, 1, input_bytes) &&
         wan_audio_bytes(hidden, group_channels, kernel_width, weight_bytes) &&
         wan_audio_bytes(kernel_width, 1, 1, scale_bytes) &&
         wan_audio_bytes(hidden, 1, 1, bias_bytes) &&
         wan_audio_bytes(frames, hidden, 1, output_bytes);
}

extern "C" void *gs_wan_audio_mps_position_plan_create(
    void *device_handle, void *input_handle, void *weight_v_handle,
    void *scale_handle, void *bias_handle, void *raw_output_handle,
    void *output_handle, int32_t frames, int32_t hidden, int32_t groups,
    int32_t kernel_width) {
  if (!device_handle || !input_handle || !weight_v_handle || !scale_handle ||
      !bias_handle || !raw_output_handle || !output_handle) return nullptr;

  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      uint64_t input_bytes = 0, weight_bytes = 0, scale_bytes = 0,
               bias_bytes = 0, output_bytes = 0;
      if (!wan_audio_position_bytes(frames, hidden, groups, kernel_width,
                                    &input_bytes, &weight_bytes, &scale_bytes,
                                    &bias_bytes, &output_bytes)) return nullptr;

      id<MTLDevice> device = (__bridge id<MTLDevice>)device_handle;
      id<MTLBuffer> input = (__bridge id<MTLBuffer>)input_handle;
      id<MTLBuffer> weightV = (__bridge id<MTLBuffer>)weight_v_handle;
      id<MTLBuffer> scale = (__bridge id<MTLBuffer>)scale_handle;
      id<MTLBuffer> bias = (__bridge id<MTLBuffer>)bias_handle;
      id<MTLBuffer> rawOutput = (__bridge id<MTLBuffer>)raw_output_handle;
      id<MTLBuffer> output = (__bridge id<MTLBuffer>)output_handle;
      id<MTLBuffer> buffers[] = {input, weightV, scale, bias, rawOutput, output};
      const uint64_t required[] = {input_bytes, weight_bytes, scale_bytes,
                                   bias_bytes, output_bytes, output_bytes};
      for (NSUInteger i = 0; i < 6; ++i) {
        if (buffers[i].device != device || buffers[i].length < required[i])
          return nullptr;
        for (NSUInteger j = 0; j < i; ++j)
          if (!wan_audio_disjoint(buffers[i], buffers[j])) return nullptr;
      }

      @try {
        CogniWanAudioMPSPositionPlan *plan =
            [CogniWanAudioMPSPositionPlan new];
        plan.device = device;
        plan.input = input;
        plan.weightV = weightV;
        plan.scale = scale;
        plan.bias = bias;
        plan.rawOutput = rawOutput;
        plan.output = output;
        plan.frames = (NSUInteger)frames;
        plan.hidden = (NSUInteger)hidden;
        plan.groups = (NSUInteger)groups;
        plan.kernelWidth = (NSUInteger)kernel_width;

        plan.graph = [[MPSGraph alloc] init];
        plan.graph.options = MPSGraphOptionsNone;
        MPSShape *inputShape = @[@1, @1, @(frames), @(hidden)];
        plan.inputTensor = [plan.graph placeholderWithShape:inputShape
                                                    dataType:MPSDataTypeFloat32
                                                        name:@"position_sequence_major_input"];
        MPSGraphTensor *swapChannelAndWidth = [plan.graph
            transposeTensor:plan.inputTensor dimension:1 withDimension:3
            name:@"position_input_swap_channel_width"];
        MPSGraphTensor *channelsFirst = [plan.graph
            transposeTensor:swapChannelAndWidth dimension:2 withDimension:3
            name:@"position_input_to_nchw"];

        const int32_t group_channels = hidden / groups;
        MPSShape *weightShape = @[@(hidden), @(group_channels), @1,
                                  @(kernel_width)];
        plan.weightTensor = [plan.graph placeholderWithShape:weightShape
                                                     dataType:MPSDataTypeFloat32
                                                         name:@"position_weight_v_oihw"];
        MPSShape *scaleShape = @[@(kernel_width)];
        plan.scaleTensor = [plan.graph placeholderWithShape:scaleShape
                                                     dataType:MPSDataTypeFloat32
                                                         name:@"position_weight_scale"];
        MPSGraphTensor *scaleOIHW = [plan.graph reshapeTensor:plan.scaleTensor
            withShape:@[@1, @1, @1, @(kernel_width)]
            name:@"position_weight_scale_oihw"];
        MPSGraphTensor *effectiveWeight = [plan.graph
            multiplicationWithPrimaryTensor:plan.weightTensor
            secondaryTensor:scaleOIHW name:@"weight_norm_scale_per_tap"];

        MPSShape *biasShape = @[@1, @(hidden), @1, @1];
        plan.biasTensor = [plan.graph placeholderWithShape:biasShape
                                                   dataType:MPSDataTypeFloat32
                                                       name:@"position_bias_nchw"];
        const NSUInteger padding = (NSUInteger)kernel_width / 2;
        MPSGraphConvolution2DOpDescriptor *descriptor =
            [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:1 strideInY:1
                dilationRateInX:1 dilationRateInY:1 groups:(NSUInteger)groups
                paddingLeft:padding paddingRight:padding
                paddingTop:0 paddingBottom:0
                paddingStyle:MPSGraphPaddingStyleExplicit
                dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        if (!descriptor) return nullptr;
        MPSGraphTensor *convolved = [plan.graph
            convolution2DWithSourceTensor:channelsFirst
            weightsTensor:effectiveWeight descriptor:descriptor
            name:@"grouped_position_conv1d"];
        MPSGraphTensor *biased = [plan.graph
            additionWithPrimaryTensor:convolved secondaryTensor:plan.biasTensor
            name:@"position_bias_after_conv"];
        // HF's SamePad removes the last of T+1 values for even K, before GELU.
        MPSGraphTensor *cropped = [plan.graph
            sliceTensor:biased dimension:3 start:0 length:frames
            name:@"position_samepad_crop_last"];
        MPSGraphTensor *sequenceMajor = [plan.graph
            transposeTensor:cropped dimension:1 withDimension:3
            name:@"position_output_to_sequence_major"];
        plan.rawTensor = [plan.graph reshapeTensor:sequenceMajor
            withShape:@[@(frames), @(hidden)] name:@"position_conv_raw"];

        MPSGraphTensor *sqrtHalf = [plan.graph constantWithScalar:
            0.7071067811865475244 shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor *erfArgument = [plan.graph
            multiplicationWithPrimaryTensor:plan.rawTensor secondaryTensor:sqrtHalf
            name:@"position_gelu_erf_argument"];
        MPSGraphTensor *erfValue = [plan.graph erfWithTensor:erfArgument
            name:@"position_gelu_erf"];
        MPSGraphTensor *one = [plan.graph constantWithScalar:1.0
            shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor *cdfSum = [plan.graph
            additionWithPrimaryTensor:erfValue secondaryTensor:one
            name:@"position_gelu_cdf_sum"];
        MPSGraphTensor *half = [plan.graph constantWithScalar:0.5
            shape:@[@1] dataType:MPSDataTypeFloat32];
        MPSGraphTensor *cdf = [plan.graph
            multiplicationWithPrimaryTensor:cdfSum secondaryTensor:half
            name:@"position_gelu_cdf"];
        plan.outputTensor = [plan.graph
            multiplicationWithPrimaryTensor:cdf secondaryTensor:plan.rawTensor
            name:@"position_gelu_exact"];

        MPSGraphShapedType *inputType = [[MPSGraphShapedType alloc]
            initWithShape:inputShape dataType:MPSDataTypeFloat32];
        MPSGraphShapedType *weightType = [[MPSGraphShapedType alloc]
            initWithShape:weightShape dataType:MPSDataTypeFloat32];
        MPSGraphShapedType *scaleType = [[MPSGraphShapedType alloc]
            initWithShape:scaleShape dataType:MPSDataTypeFloat32];
        MPSGraphShapedType *biasType = [[MPSGraphShapedType alloc]
            initWithShape:biasShape dataType:MPSDataTypeFloat32];
        if (!inputType || !weightType || !scaleType || !biasType ||
            !plan.rawTensor || !plan.outputTensor) return nullptr;
        MPSGraphTensorShapedTypeDictionary *feedTypes = @{
          plan.inputTensor : inputType,
          plan.weightTensor : weightType,
          plan.scaleTensor : scaleType,
          plan.biasTensor : biasType,
        };
        plan.executable = [plan.graph
            compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                        feeds:feedTypes
                targetTensors:@[plan.rawTensor, plan.outputTensor]
             targetOperations:nil compilationDescriptor:nil];
        if (!plan.executable) return nullptr;
        plan.executable.options = MPSGraphOptionsNone;

        plan.inputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input shape:inputShape dataType:MPSDataTypeFloat32];
        plan.weightData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:weightV shape:weightShape dataType:MPSDataTypeFloat32];
        plan.scaleData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:scale shape:scaleShape dataType:MPSDataTypeFloat32];
        plan.biasData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bias shape:biasShape dataType:MPSDataTypeFloat32];
        MPSShape *outputShape = @[@(frames), @(hidden)];
        plan.rawOutputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:rawOutput shape:outputShape dataType:MPSDataTypeFloat32];
        plan.outputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:output shape:outputShape dataType:MPSDataTypeFloat32];
        if (!plan.inputData || !plan.weightData || !plan.scaleData ||
            !plan.biasData || !plan.rawOutputData || !plan.outputData ||
            plan.executable.feedTensors.count != 4 ||
            plan.executable.targetTensors.count != 2 ||
            plan.executable.targetTensors[0] != plan.rawTensor ||
            plan.executable.targetTensors[1] != plan.outputTensor) return nullptr;

        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feedData = @{
          plan.inputTensor : plan.inputData,
          plan.weightTensor : plan.weightData,
          plan.scaleTensor : plan.scaleData,
          plan.biasTensor : plan.biasData,
        };
        NSMutableArray<MPSGraphTensorData *> *orderedInputs =
            [NSMutableArray arrayWithCapacity:4];
        for (MPSGraphTensor *feedTensor in plan.executable.feedTensors) {
          MPSGraphTensorData *feedValue = feedData[feedTensor];
          if (!feedValue) return nullptr;
          [orderedInputs addObject:feedValue];
        }
        plan.inputArray = [orderedInputs copy];
        plan.outputArray = @[plan.rawOutputData, plan.outputData];
        plan.execution = [MPSGraphExecutableExecutionDescriptor new];
        plan.execution.waitUntilCompleted = NO;
        if (!plan.inputArray || !plan.outputArray || !plan.execution)
          return nullptr;
        return (__bridge_retained void *)plan;
      } @catch (NSException *exception) {
        (void)exception;
        return nullptr;
      }
    }
  }
  return nullptr;
}

extern "C" int32_t gs_wan_audio_mps_position_encode(
    void *plan_handle, void *command_handle) {
  if (!plan_handle || !command_handle) return -1;
  if (@available(macOS 15.0, *)) {
    @autoreleasepool {
      CogniWanAudioMPSPositionPlan *plan =
          (__bridge CogniWanAudioMPSPositionPlan *)plan_handle;
      MPSCommandBuffer *command = (__bridge MPSCommandBuffer *)command_handle;
      id<MTLCommandBuffer> root = command.rootCommandBuffer;
      if (!root || root.commandQueue.device != plan.device ||
          root.status >= MTLCommandBufferStatusCommitted) return -2;
      @try {
        NSArray<MPSGraphTensorData *> *results =
            [plan.executable encodeToCommandBuffer:command
                                       inputsArray:plan.inputArray
                                      resultsArray:plan.outputArray
                               executionDescriptor:plan.execution];
        if (results.count != 2) return -4;
      } @catch (NSException *exception) {
        (void)exception;
        return -5;
      }
      return 0;
    }
  }
  return -6;
}

extern "C" void gs_wan_audio_mps_position_plan_release(void *plan_handle) {
  if (plan_handle) {
    id plan = (__bridge_transfer id)plan_handle;
    (void)plan;
  }
}
