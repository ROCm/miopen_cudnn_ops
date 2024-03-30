#ifndef __RNN_H
#define __RNN_H

#include <cudnn.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

template <typename T>
class RNN_IMPL {
   public:
    void *x;
    void *hx;
    void *cx;

    void *dx;
    void *dhx;
    void *dcx;

    void *y;
    void *hy;
    void *cy;

    void *dy;
    void *dhy;
    void *dcy;

    // data type (0-FP32, 2-FP16)
    cudnnDataType_t dataType;
    // math precision (0-FP32, 2-FP16)
    cudnnDataType_t mathPrecision;
    // math type (0-default, 1-tensor op math, 2-tensor op math with conversion)
    cudnnMathType_t mathType;
    // recurrence pattern (0-unidirectional, 1-bidirectional)
    cudnnDirectionMode_t recurrencePattern;
    // recurrence algorithm (0-standard, 1-persist static, 2-persist dynamics)
    cudnnRNNAlgo_t rnnAlgorithm;
    // bias mode (0-no bias, 1-inp bias, 2-double bias, 3-rec bias)
    cudnnRNNBiasMode_t rnnBiasMode;
    // cell type (0-RELU, 1-TANH, 2-LSTM, 3-GRU)
    cudnnRNNMode_t rnnCellMode;
    // input mode (0-linear input, 1-skip input)
    cudnnRNNInputMode_t rnnInputMode;

    // dropout rate
    float dropout;
    // hidden size
    int hiddenSize;
    // input vector size
    int inputSize;
    // max miniBatch size
    int miniBatch;
    // number of layers
    int numLayers;
    // LSTM cell output size after the recurrent projection
    int outputSize;
    // sequence length
    int seqLength;


    // number of warm-up iterations
    int warmupIterations;
    // number of measurement iterations
    int measureIterations;

    std::mt19937 eng;
    std::uniform_real_distribution<> dist;

    cudnnHandle_t cudnnHandle;
    cudnnRNNDescriptor_t rnnDesc;

    cudnnRNNDataLayout_t rnnDataLayout = CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED;
    cudnnForwardMode_t fwdMode = CUDNN_FWD_MODE_TRAINING;
    cudnnWgradMode_t wgradMode = CUDNN_WGRAD_MODE_ADD;

    std::vector<int> seqLengthArray;
    int* devSeqLengthArray;

    size_t weightSpaceSize;
    size_t workSpaceSize;
    size_t reserveSpaceSize;

    void* weightSpace = NULL;
    void* dweightSpace = NULL;
    void* workSpace = NULL;
    void* reserveSpace = NULL;

    RNN_IMPL<T>() :
        x  (NULL),
        hx (NULL),
        cx (NULL),
        dx (NULL),
        dhx(NULL),
        dcx(NULL),
        y  (NULL),
        hy (NULL),
        cy (NULL),
        dy (NULL),
        dhy(NULL),
        dcy(NULL),
        warmupIterations(3),
        measureIterations(10),
        cudnnHandle(NULL),
        rnnDesc(NULL),
        rnnDataLayout(CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED),
        fwdMode(CUDNN_FWD_MODE_TRAINING),
        wgradMode(CUDNN_WGRAD_MODE_ADD),
        devSeqLengthArray(NULL),
        weightSpace(NULL),
        dweightSpace(NULL),
        workSpace(NULL),
        reserveSpace(NULL)
        {
            std::seed_seq seed{0};
            eng = std::mt19937{seed};
            dist = std::uniform_real_distribution<>(0, 1);

            // create cudnn context
            cudnnCreate(&cudnnHandle);

            // create rnn descriptor
            cudnnCreateRNNDescriptor(&rnnDesc);
        };

    ~RNN_IMPL<T>()
    {
        cudnnDestroy(cudnnHandle);
    }

    // initialize the device memory with values from a uniform distribution between 0 and 1.
    void real(void* dst, const size_t& size)
    {
        std::vector<T> values(size);

        std::generate(values.begin(), values.end(), [&]{ return dist(eng); });

        cudaMemcpy(dst, &values[0], size * sizeof(T), cudaMemcpyHostToDevice);
    }

    // initialize the device memory with 1.
    void ones(void* dst, const size_t& size)
    {
        std::vector<T> values(size, 1.0);

        cudaMemcpy(dst, &values[0], size * sizeof(T), cudaMemcpyHostToDevice);
    }

    void validate();

    void init();

    void run(float& timeForward,
    float& timeBackwardData,
    float& timeBackwardWeights);

    float forward();
    float backward_data();
    float backward_filter();

    enum errors_t
    {
        NO_ERROR           = 0,
        DATA_TYPE          = 2^0,
        MATH_PRECISION     = 2^1,
        MATH_TYPE          = 2^2,
        RECURRENCE_PATTERN = 2^3,
        RNN_ALGORITHM      = 2^4,
        RNN_BIAS_MODE      = 2^5,
        RNN_CELL_MODE      = 2^6,
        RNN_INPUT_MODE     = 2^7,
        INPUT_SIZE         = 2^8,
        OUTPUT_SIZE        = 2^9,
    };

    // data type (0-FP32, 2-FP16)
    errors_t check(const cudnnDataType_t& dataType)
    {
        switch(dataType)
        {
            case CUDNN_DATA_FLOAT: return NO_ERROR;
            case CUDNN_DATA_HALF:  return NO_ERROR;
            default: return DATA_TYPE;
        }
    }

    // math type (0-default, 1-tensor op math, 2-tensor op math with conversion)
    errors_t check(const cudnnMathType_t& mathType)
    {
        switch(mathType)
        {
            case CUDNN_DEFAULT_MATH:                    return NO_ERROR;
            case CUDNN_TENSOR_OP_MATH:                  return NO_ERROR;
            case CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION: return NO_ERROR;
            default: return MATH_TYPE;
        }
    }

    // recurrence pattern (0-unidirectional, 1-bidirectional)
    errors_t check(const cudnnDirectionMode_t& recurrencePattern)
    {
        switch(recurrencePattern)
        {
            case CUDNN_UNIDIRECTIONAL: return NO_ERROR;
            case CUDNN_BIDIRECTIONAL:  return NO_ERROR;
            default: return RECURRENCE_PATTERN;
        }
    }

    // recurrence algorithm (0-standard, 1-persist static, 2-persist dynamics)
    errors_t check(const cudnnRNNAlgo_t& rnnAlgorithm)
    {
        switch(rnnAlgorithm)
        {
            case CUDNN_RNN_ALGO_STANDARD:        return NO_ERROR;
            case CUDNN_RNN_ALGO_PERSIST_STATIC:  return NO_ERROR;
            case CUDNN_RNN_ALGO_PERSIST_DYNAMIC: return NO_ERROR;
            default: return RNN_ALGORITHM;
        }
    }

    // bias mode (0-no bias, 1-inp bias, 2-double bias, 3-rec bias)
    errors_t check(const cudnnRNNBiasMode_t& rnnBiasMode)
    {
        switch(rnnBiasMode)
        {
            case CUDNN_RNN_NO_BIAS:         return NO_ERROR;
            case CUDNN_RNN_SINGLE_INP_BIAS: return NO_ERROR;
            case CUDNN_RNN_DOUBLE_BIAS:     return NO_ERROR;
            case CUDNN_RNN_SINGLE_REC_BIAS: return NO_ERROR;
            default: return RNN_BIAS_MODE;
        }
    }

    // cell type (0-RELU, 1-TANH, 2-LSTM, 3-GRU)
    errors_t check(const cudnnRNNMode_t& rnnCellMode)
    {
        switch(rnnCellMode)
        {
            case CUDNN_RNN_RELU: return NO_ERROR;
            case CUDNN_RNN_TANH: return NO_ERROR;
            case CUDNN_LSTM:     return NO_ERROR;
            case CUDNN_GRU:      return NO_ERROR;
            default: return RNN_CELL_MODE;
        }
    }

    // input mode (0-linear input, 1-skip input)
    errors_t check(const cudnnRNNInputMode_t& rnnInputMode)
    {
        switch(rnnInputMode)
        {
            case CUDNN_LINEAR_INPUT: return NO_ERROR;
            case CUDNN_SKIP_INPUT:   return NO_ERROR;
            default: return RNN_INPUT_MODE;
        }
    }

    errors_t check(const cudnnDataType_t& dataType,
        const cudnnDataType_t& mathPrecision)
    {
        switch(dataType)
        {
            case CUDNN_DATA_FLOAT:
            {
                switch(mathPrecision)
                {
                    case CUDNN_DATA_FLOAT: return NO_ERROR;
                    default: return MATH_PRECISION;
                }
            }
            case CUDNN_DATA_HALF:
            {
                switch(mathPrecision)
                {
                    case CUDNN_DATA_FLOAT: return NO_ERROR;
                    case CUDNN_DATA_HALF:  return NO_ERROR;
                    default: return MATH_PRECISION;
                }
            }
            default: return NO_ERROR;
        }
    }

    errors_t check(const cudnnDataType_t& dataType,
        const cudnnMathType_t& mathType)
    {
        switch(dataType)
        {
            case CUDNN_DATA_FLOAT:
            {
                switch(mathType)
                {
                    case CUDNN_DEFAULT_MATH:                    return NO_ERROR;
                    case CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION: return NO_ERROR;
                    default: return MATH_TYPE;
                }
            }
            default: return NO_ERROR;
        }
    }

    errors_t check(const cudnnRNNInputMode_t& rnnInputMode,
        const int& inputSize, const int& hiddenSize)
    {
        if (rnnInputMode != CUDNN_SKIP_INPUT) return NO_ERROR;
        if (inputSize == hiddenSize)          return NO_ERROR;
        return INPUT_SIZE;
    }

    errors_t check(const int& outputSize, const int& hiddenSize)
    {
        if (outputSize <= hiddenSize) return NO_ERROR;
        return OUTPUT_SIZE;
    }

    // get cell mode from string
    cudnnRNNMode_t get_cellMode(const std::string& input)
    {
        if (input == "relu")
            return CUDNN_RNN_RELU;
        if (input == "tanh")
            return CUDNN_RNN_TANH;
        if (input == "lstm")
            return CUDNN_LSTM;
        if (input == "gru")
            return CUDNN_GRU;

        // this seems to be the default value in MIOpenDriver
        return CUDNN_RNN_TANH;
    }

    static int BidirectionalScale(const cudnnDirectionMode_t& recurrencePattern)
    {
        return (recurrencePattern == CUDNN_BIDIRECTIONAL ? 2 : 1);
    };

    static size_t InputTensorSize(const int& seqLength,
        const int& miniBatch, const int& inputSize)
    {
        return seqLength * miniBatch * inputSize;
    };

    static size_t OutputTensorSize(const int& seqLength, const int& miniBatch,
        const int& hiddenSize, const cudnnDirectionMode_t& recurrencePattern)
    {
        int bidirectionalScale = BidirectionalScale(recurrencePattern);
        return seqLength * miniBatch * hiddenSize * bidirectionalScale;
    };

    static size_t HiddenTensorSize(const int& numLayers, const int& miniBatch,
        const int& hiddenSize, const cudnnDirectionMode_t& recurrencePattern)
    {
        int bidirectionalScale = BidirectionalScale(recurrencePattern);
        return numLayers * miniBatch * hiddenSize * bidirectionalScale;
    };

    static int NumLinearLayers(const cudnnRNNMode_t& rnnCellMode)
    {
        int output = 0;
        if (rnnCellMode == CUDNN_RNN_RELU || rnnCellMode == CUDNN_RNN_TANH)
        {
            output = 2;
        }
        else if (rnnCellMode == CUDNN_LSTM)
        {
            output = 8;
        }
        else if (rnnCellMode == CUDNN_GRU)
        {
            output = 6;
        }

        return output;
    };

    static const char* to_str(const cudnnDataType_t& value)
    {
        switch(value)
        {
            case CUDNN_DATA_FLOAT:              return "CUDNN_DATA_FLOAT";
            case CUDNN_DATA_DOUBLE:             return "CUDNN_DATA_DOUBLE";
            case CUDNN_DATA_HALF:               return "CUDNN_DATA_HALF";
            case CUDNN_DATA_INT8:               return "CUDNN_DATA_INT8";
            case CUDNN_DATA_INT32:              return "CUDNN_DATA_INT32";
            case CUDNN_DATA_INT8x4:             return "CUDNN_DATA_INT8x4";
            case CUDNN_DATA_UINT8:              return "CUDNN_DATA_UINT8";
            case CUDNN_DATA_UINT8x4:            return "CUDNN_DATA_UINT8x4";
            case CUDNN_DATA_INT8x32:            return "CUDNN_DATA_INT8x32";
            case CUDNN_DATA_BFLOAT16:           return "CUDNN_DATA_BFLOAT16";
            case CUDNN_DATA_INT64:              return "CUDNN_DATA_INT64";
            case CUDNN_DATA_BOOLEAN:            return "CUDNN_DATA_BOOLEAN";
            case CUDNN_DATA_FP8_E4M3:           return "CUDNN_DATA_FP8_E4M3";
            case CUDNN_DATA_FP8_E5M2:           return "CUDNN_DATA_FP8_E5M2";
            case CUDNN_DATA_FAST_FLOAT_FOR_FP8: return "CUDNN_DATA_FAST_FLOAT_FOR_FP8";
            default: return "UNDEFINED";
        }
    }

    static const char* to_str(const cudnnRNNInputMode_t& value)
    {
        switch(value)
        {
            case CUDNN_LINEAR_INPUT:            return "CUDNN_LINEAR_INPUT";
            case CUDNN_SKIP_INPUT:              return "CUDNN_SKIP_INPUT";
            default: return "UNDEFINED";
        }
    }

    static const char* to_str(const cudnnDirectionMode_t& value)
    {
        switch(value)
        {
            case CUDNN_UNIDIRECTIONAL:          return "CUDNN_LINEAR_INPUT";
            case CUDNN_BIDIRECTIONAL:           return "CUDNN_SKIP_INPUT";
            default: return "UNDEFINED";
        }
    }

    static const char* to_str(const cudnnRNNMode_t& value)
    {
        switch(value)
        {
            case CUDNN_RNN_RELU:                return "CUDNN_RNN_RELU";
            case CUDNN_RNN_TANH:                return "CUDNN_RNN_TANH";
            case CUDNN_LSTM:                    return "CUDNN_LSTM";
            case CUDNN_GRU:                     return "CUDNN_GRU";
            default: return "UNDEFINED";
        }
    }

    static const char* to_str(const cudnnRNNBiasMode_t& value)
    {
        switch(value)
        {
            case CUDNN_RNN_NO_BIAS:             return "CUDNN_RNN_NO_BIAS";
            case CUDNN_RNN_SINGLE_INP_BIAS:     return "CUDNN_RNN_SINGLE_INP_BIAS";
            case CUDNN_RNN_DOUBLE_BIAS:         return "CUDNN_RNN_DOUBLE_BIAS";
            case CUDNN_RNN_SINGLE_REC_BIAS:     return "CUDNN_RNN_SINGLE_REC_BIAS";
            default: return "UNDEFINED";
        }
    }

    static const char* to_str(const cudnnRNNAlgo_t& value)
    {
        switch(value)
        {
            case CUDNN_RNN_ALGO_STANDARD:               return "CUDNN_RNN_ALGO_STANDARD";
            case CUDNN_RNN_ALGO_PERSIST_STATIC:         return "CUDNN_RNN_ALGO_PERSIST_STATIC";
            case CUDNN_RNN_ALGO_PERSIST_DYNAMIC:        return "CUDNN_RNN_ALGO_PERSIST_DYNAMIC";
            case CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H: return "CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H";
            case CUDNN_RNN_ALGO_COUNT:                  return "CUDNN_RNN_ALGO_COUNT";
            default: return "UNDEFINED";
        }
    }

    static const char* to_str(const cudnnMathType_t& value)
    {
        switch(value)
        {
            case CUDNN_DEFAULT_MATH:                    return "CUDNN_DEFAULT_MATH";
            case CUDNN_TENSOR_OP_MATH:                  return "CUDNN_TENSOR_OP_MATH";
            case CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION: return "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION";
            case CUDNN_FMA_MATH:                        return "CUDNN_FMA_MATH";
            default: return "UNDEFINED";
        }
    }

    static void print_info(const std::string& str1,
    const double& value = -1.0, const std::string& str2 = "")
    {
        std::cout << std::right << std::setw(20) << str1  << ": ";
        if (value != -1)
        {
            std::cout << std::right << std::setw(8) << std::setprecision(4) << value << " ";
            std::cout << std::left  << std::setw(30) << str2  << " ";
        }
        std::cout << std::endl;
    }
};

template <typename T>
void RNN_IMPL<T>::validate()
{
    int error = 0;
    error |= check(dataType);
    error |= check(dataType, mathPrecision);
    error |= check(dataType, mathType);
    // error |= check(mathPrecision);
    // error |= check(mathType);
    error |= check(recurrencePattern);
    error |= check(rnnAlgorithm);
    error |= check(rnnBiasMode);
    error |= check(rnnCellMode);
    error |= check(rnnInputMode);
    error |= check(rnnInputMode, inputSize, hiddenSize);
    error |= check(outputSize, hiddenSize);


    if (error & DATA_TYPE)          std::cout << "dataType must be 0 (FP32) or 2 (FP16), as in cudnnDataType_t." << std::endl;
    if (error & MATH_PRECISION)     std::cout << "mathPrecision does not match dataType." << std::endl;
    if (error & MATH_TYPE)          std::cout << "mathType does not match dataType." << std::endl;
    if (error & RECURRENCE_PATTERN) std::cout << "recurrencePattern must be 0 (unidirectional) or  1 (bidirectional), as in cudnnDirectionMode_t." << std::endl;
    if (error & RNN_ALGORITHM)      std::cout << "rnnAlgorithm  must be 0 (standard), 1 (persist static) or 2 (persist dynamics), as in cudnnRNNAlgo_t." << std::endl;
    if (error & RNN_BIAS_MODE)      std::cout << "rnnBiasMode must be 0 (no bias), 1 (inp bias), 2 (double bias) or  3 (rec bias), as in cudnnRNNBiasMode_t." << std::endl;
    if (error & RNN_CELL_MODE)      std::cout << "rnnCellMode must be 0 (RELU), 1 (TANH), 2 (LSTM) or 3 (GRU), as in cudnnRNNMode_t." << std::endl;
    if (error & RNN_INPUT_MODE)     std::cout << "input mode 0 (linear input) or 1 (skip input), as in cudnnRNNInputMode_t." << std::endl;
    if (error & INPUT_SIZE)         std::cout << "inputSize must be equal to hiddenSize." << std::endl;
    if (error & OUTPUT_SIZE)        std::cout << "outputSize must be less then or equal to the hiddenSize." << std::endl;
    if (error != 0) exit(error);

    print_info("Arguments");
    print_info("seqLength", seqLength);
    print_info("numLayers", numLayers);
    print_info("inputSize", inputSize);
    print_info("hiddenSize", hiddenSize);
    print_info("outputSize", outputSize);
    print_info("miniBatch", miniBatch);
    print_info("rnnInputMode", rnnInputMode, to_str(rnnInputMode));
    print_info("recurrencePattern", recurrencePattern, to_str(recurrencePattern));
    print_info("rnnCellMode", rnnCellMode, to_str(rnnCellMode));
    print_info("rnnBiasMode", rnnBiasMode, to_str(rnnBiasMode));
    print_info("rnnAlgorithm", rnnAlgorithm, to_str(rnnAlgorithm));
    print_info("mathPrecision", mathPrecision, to_str(mathPrecision));
    print_info("mathType", mathType, to_str(mathType));
    print_info("dataType", dataType, to_str(dataType));
    print_info("dropout", dropout);
}

// TODO run init before rnn execution, outside run(...)
template <typename T>
void RNN_IMPL<T>::init()
{
    int bidirectionalScale = BidirectionalScale(recurrencePattern);
    size_t inputTensorSize  = InputTensorSize(seqLength, miniBatch, inputSize);
    size_t outputTensorSize = OutputTensorSize(seqLength, miniBatch,
        hiddenSize, recurrencePattern);
    size_t hiddenTensorSize = HiddenTensorSize(numLayers, miniBatch,
        hiddenSize, recurrencePattern);
    int numLinearLayers = NumLinearLayers(rnnCellMode);

    // TODO: absolutely need to make sure this is initialized before most everything else
    // Memory allocation for seqLengthArray on the host and device
    seqLengthArray = std::vector<int>(miniBatch);
    int* devSeqLengthArray = NULL;
    for (int i = 0; i < miniBatch; i++)
        seqLengthArray[i] = seqLength;
    cudaMalloc((void**)&devSeqLengthArray, miniBatch * sizeof(int));
    cudaMemcpy(devSeqLengthArray, &seqLengthArray[0], miniBatch * sizeof(int),
        cudaMemcpyHostToDevice);

    // Initialize inputs
    ones(x, inputTensorSize);
    ones(hx, hiddenTensorSize);
    ones(cx, hiddenTensorSize);
    ones(dy, outputTensorSize);
    ones(dhy, hiddenTensorSize);
    ones(dcy, hiddenTensorSize);

    // initialize weights
    cudnnTensorDescriptor_t wDesc;
    cudnnTensorDescriptor_t bDesc;

    cudnnCreateTensorDescriptor(&wDesc);
    cudnnCreateTensorDescriptor(&bDesc);

    for (int layer = 0; layer < numLayers * bidirectionalScale; layer++)
    {
        for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++)
        {
            cudnnDataType_t dataTypeTemp;
            int nbDims = 0;
            int dim[3], stride[3];
            T* linLayerMat  = NULL;
            T* linLayerBias = NULL;

            cudnnGetRNNWeightParams(cudnnHandle, rnnDesc, layer,
                weightSpaceSize, weightSpace, linLayerID, wDesc,
                (void**)&linLayerMat, bDesc, (void**)&linLayerBias);

            if (linLayerMat)
            {
                // initialize using a uniform real distribution betwen 0 and 1
                cudnnGetTensorNdDescriptor(wDesc, 3, &dataTypeTemp, &nbDims,
                    dim, stride);
                real(linLayerMat, dim[0] * dim[1] * dim[2]);
            }

            if (linLayerBias)
            {
                // initialize with 1.0
                cudnnGetTensorNdDescriptor(bDesc, 3, &dataTypeTemp, &nbDims,
                    dim, stride);
                ones(linLayerBias, dim[0] * dim[1] * dim[2]);
            }
        }
    }

    cudnnDestroyTensorDescriptor(wDesc);
    cudnnDestroyTensorDescriptor(bDesc);
}

template <typename T>
void RNN_IMPL<T>::run(float& timeForward,
    float& timeBackwardData,
    float& timeBackwardWeights)
{
    // Dimensions for hidden state tensors
    int bidirectionalScale = BidirectionalScale(recurrencePattern);
    int dimHidden[3] = {numLayers * bidirectionalScale, miniBatch, hiddenSize};
    int strideHidden[3] = {dimHidden[1] * dimHidden[2], dimHidden[2], 1};

    size_t inputTensorSize = InputTensorSize(seqLength, miniBatch, inputSize);
    size_t outputTensorSize = OutputTensorSize(seqLength, miniBatch,
        hiddenSize, recurrencePattern);
    size_t hiddenTensorSize = HiddenTensorSize(numLayers, miniBatch,
        hiddenSize, recurrencePattern);

    T paddingFill = 0.0;

    // Dropout descriptor parameters
    unsigned long long seed = 1337ull;
    size_t stateSize;
    void* states = NULL;

    // Profiling parameters
    cudaEvent_t start;
    cudaEvent_t stop;

    cudnnRNNDataDescriptor_t xDesc = NULL;
    cudnnRNNDataDescriptor_t yDesc = NULL;

    cudnnTensorDescriptor_t hDesc = NULL;
    cudnnTensorDescriptor_t cDesc = NULL;

    cudnnDropoutDescriptor_t dropoutDesc = NULL;

    // Initialize all the data
    init();

    // Memory allocation. hx, cx, dhx, dcx, hy, cy, dhy and dcy can be NULL.
    cudaMalloc((void**)&x,  inputTensorSize  * sizeof(T));
    cudaMalloc((void**)&y,  outputTensorSize * sizeof(T));
    cudaMalloc((void**)&dx, inputTensorSize  * sizeof(T));
    cudaMalloc((void**)&dy, outputTensorSize * sizeof(T));

    cudaMalloc((void**)&hx,  hiddenTensorSize * sizeof(T));
    cudaMalloc((void**)&cx,  hiddenTensorSize * sizeof(T));
    cudaMalloc((void**)&hy,  hiddenTensorSize * sizeof(T));
    cudaMalloc((void**)&cy,  hiddenTensorSize * sizeof(T));
    cudaMalloc((void**)&dhx, hiddenTensorSize * sizeof(T));
    cudaMalloc((void**)&dcx, hiddenTensorSize * sizeof(T));
    cudaMalloc((void**)&dhy, hiddenTensorSize * sizeof(T));
    cudaMalloc((void**)&dcy, hiddenTensorSize * sizeof(T));

    // create/set RNN x data descriptors
    cudnnCreateRNNDataDescriptor(&xDesc);
    cudnnSetRNNDataDescriptor(xDesc, dataType, rnnDataLayout, seqLength,
        miniBatch, inputSize, &seqLengthArray[0], &paddingFill);

    // create/set RNN y data descriptors
    cudnnCreateRNNDataDescriptor(&yDesc);
    cudnnSetRNNDataDescriptor(yDesc, dataType, rnnDataLayout, seqLength,
        miniBatch, hiddenSize * bidirectionalScale, &seqLengthArray[0], &paddingFill);

    cudnnCreateTensorDescriptor(&hDesc);
    cudnnSetTensorNdDescriptor(hDesc, dataType, 3, dimHidden, strideHidden);

    cudnnCreateTensorDescriptor(&cDesc);
    cudnnSetTensorNdDescriptor(cDesc, dataType, 3, dimHidden, strideHidden);

    // Set up the dropout descriptor (needed for the RNN descriptor)
    cudnnCreateDropoutDescriptor(&dropoutDesc);
    cudnnDropoutGetStatesSize(cudnnHandle, &stateSize);
    cudaMalloc(&states, stateSize);
    cudnnSetDropoutDescriptor(dropoutDesc, cudnnHandle, dropout, states,
        stateSize, seed);

    // Set up the RNN descriptor
    cudnnSetRNNDescriptor_v8(rnnDesc, rnnAlgorithm, rnnCellMode, rnnBiasMode,
        recurrencePattern, rnnInputMode, dataType, mathPrecision, mathType,
        inputSize, hiddenSize, outputSize, numLayers, dropoutDesc, 0);

    // Set up weights and bias parameters
    cudnnGetRNNWeightSpaceSize(cudnnHandle, rnnDesc, &weightSpaceSize);
    cudaMalloc((void**)&weightSpace, weightSpaceSize);
    cudaMalloc((void**)&dweightSpace, weightSpaceSize);

    // Set up work space and reserved memory
    cudnnGetRNNTempSpaceSizes(cudnnHandle, rnnDesc, fwdMode, xDesc,
        &workSpaceSize, &reserveSpaceSize);
    cudaMalloc((void**)&workSpace, workSpaceSize);
    cudaMalloc((void**)&reserveSpace, reserveSpaceSize);

    // dynamic persistent RNN plan
    if (rnnAlgorithm == CUDNN_RNN_ALGO_PERSIST_DYNAMIC)
        cudnnBuildRNNDynamic(cudnnHandle, rnnDesc, miniBatch);

    cudaDeviceSynchronize();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // timeForward = forward();
    for (int i = 0; i < warmupIterations + measureIterations; i++)
    {
        float elapsedTime = 0.0;
        cudaEventRecord(start);

        cudnnRNNForward(cudnnHandle, rnnDesc, fwdMode, devSeqLengthArray,
            xDesc, x, yDesc, y, hDesc, hx, hy, cDesc, cx, cy, weightSpaceSize,
            weightSpace, workSpaceSize, workSpace, reserveSpaceSize,
            reserveSpace);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeForward, start, stop);

        if (i > warmupIterations)
            timeForward += elapsedTime;
    }
    timeForward /= measureIterations;

    // backward data
    for (int i = 0; i < warmupIterations + measureIterations; i++)
    {
        float elapsedTime = 0.0;
        cudaEventRecord(start);

        cudnnRNNBackwardData_v8(cudnnHandle, rnnDesc, devSeqLengthArray, yDesc,
            y, dy, xDesc, dx, hDesc, hx, dhy, dhx, cDesc, cx, dcy, dcx,
            weightSpaceSize, weightSpace, workSpaceSize, workSpace,
            reserveSpaceSize, reserveSpace);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeBackwardData, start, stop);

        if (i > warmupIterations)
            timeBackwardData += elapsedTime;
    }
    timeBackwardData /= measureIterations;

    // backward weights
    for (int i = 0; i < warmupIterations + measureIterations; i++)
    {
        float elapsedTime = 0.0;

        cudaEventRecord(start);

        cudaMemset(dweightSpace, 0, weightSpaceSize);

        cudnnRNNBackwardWeights_v8(cudnnHandle, rnnDesc, wgradMode,
            devSeqLengthArray, xDesc, x, hDesc, hx, yDesc, y, weightSpaceSize,
            dweightSpace, workSpaceSize, workSpace, reserveSpaceSize,
            reserveSpace);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&timeBackwardWeights, start, stop);

        if (i > warmupIterations)
            timeBackwardWeights += elapsedTime;
    }
    timeBackwardWeights /= measureIterations;

    cudaDeviceSynchronize();

    // clean-up
    cudaFree(x);
    cudaFree(hx);
    cudaFree(cx);
    cudaFree(y);
    cudaFree(hy);
    cudaFree(cy);
    cudaFree(dx);
    cudaFree(dhx);
    cudaFree(dcx);
    cudaFree(dy);
    cudaFree(dhy);
    cudaFree(dcy);
    cudaFree(workSpace);
    cudaFree(reserveSpace);
    cudaFree(weightSpace);
    cudaFree(dweightSpace);
    cudaFree(states);
    cudaFree(devSeqLengthArray);

    cudnnDestroyRNNDataDescriptor(xDesc);
    cudnnDestroyRNNDataDescriptor(yDesc);

    cudnnDestroyTensorDescriptor(hDesc);
    cudnnDestroyTensorDescriptor(cDesc);

    cudnnDestroyDropoutDescriptor(dropoutDesc);
    cudnnDestroyRNNDescriptor(rnnDesc);

}

template <typename T>
float RNN_IMPL<T>::forward()
{
    float timeForwrd = 0.0;

    // cudaEvent_t start;
    // cudaEvent_t stop;

    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // for (int i = 0; i < warmupIterations + measureIterations; i++)
    // {
    //     float elapsedTime = 0.0;
    //     cudaEventRecord(start);

    //     cudnnRNNForward(cudnnHandle, rnnDesc, fwdMode, devSeqLengthArray,
    //         xDesc, x, yDesc, y, hDesc, hx, hy, cDesc, cx, cy, weightSpaceSize,
    //         weightSpace, workSpaceSize, workSpace, reserveSpaceSize,
    //         reserveSpace);

    //     cudaEventRecord(stop);
    //     cudaEventSynchronize(stop);
    //     cudaEventElapsedTime(&timeForward, start, stop);

    //     if (i > warmupIterations)
    //         timeForward += elapsedTime;
    // }
    // timeForward /= measureIterations;

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);

    return timeForwrd;
}

template <typename T>
float RNN_IMPL<T>::backward_data()
{
    float timeBackwardData = 0.0;

    return timeBackwardData;
}

template <typename T>
float RNN_IMPL<T>::backward_filter()
{
    float timeBackwardWeights = 0.0;

    return timeBackwardWeights;
}

#endif
