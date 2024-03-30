#include "operator.hpp"
#include <vector>
#include <iostream>
#include <iomanip>

op_rnn_cudnn::op_rnn_cudnn(void * desc) : op_rnn(desc){}
op_rnn_cudnn::~op_rnn_cudnn(){}

void op_rnn_cudnn::tune_op()
{
    if (dataType == CUDNN_DATA_FLOAT)
    {
        rnn = new RNN_IMPL<float>();

        rnn->dataType            = dataType;
        rnn->seqLength           = seqLength;
        rnn->numLayers           = numLayers;
        rnn->inputSize           = inputSize;
        rnn->hiddenSize          = hiddenSize;
        rnn->outputSize          = outputSize;
        rnn->miniBatch           = miniBatch;
        rnn->rnnInputMode        = rnnInputMode;
        rnn->recurrencePattern   = recurrencePattern;
        rnn->rnnCellMode         = rnnCellMode;
        rnn->rnnBiasMode         = rnnBiasMode;
        rnn->rnnAlgorithm        = rnnAlgorithm;
        rnn->mathPrecision       = mathPrecision;
        rnn->mathType            = mathType;
        rnn->dropout             = dropout;
        rnn->warmupIterations    = warmupIterations;
        rnn->measureIterations   = measureIterations;

        rnn->init();
        rnn->validate();
    }
    else if (dataType == CUDNN_DATA_HALF)
    {
        rnnfp16 = new RNN_IMPL<half>();

        rnnfp16->dataType            = dataType;
        rnnfp16->seqLength           = seqLength;
        rnnfp16->numLayers           = numLayers;
        rnnfp16->inputSize           = inputSize;
        rnnfp16->hiddenSize          = hiddenSize;
        rnnfp16->outputSize          = outputSize;
        rnnfp16->miniBatch           = miniBatch;
        rnnfp16->rnnInputMode        = rnnInputMode;
        rnnfp16->recurrencePattern   = recurrencePattern;
        rnnfp16->rnnCellMode         = rnnCellMode;
        rnnfp16->rnnBiasMode         = rnnBiasMode;
        rnnfp16->rnnAlgorithm        = rnnAlgorithm;
        rnnfp16->mathPrecision       = mathPrecision;
        rnnfp16->mathType            = mathType;
        rnnfp16->dropout             = dropout;
        rnnfp16->warmupIterations    = warmupIterations;
        rnnfp16->measureIterations   = measureIterations;

        rnnfp16->init();
        rnnfp16->validate();
    }
}

void op_rnn_cudnn::forward()
{
    if (dataType == CUDNN_DATA_FLOAT)
    {
        timeForward = rnn->forward();
    }
    else if (dataType == CUDNN_DATA_HALF)
    {
        timeForward = rnnfp16->forward();
    }
}

void op_rnn_cudnn::backward_data()
{
    if (dataType == CUDNN_DATA_FLOAT)
    {
        timeBackwardData = rnn->backward_data();
    }
    else if (dataType == CUDNN_DATA_HALF)
    {
        timeBackwardData = rnnfp16->backward_data();
    }
}

void op_rnn_cudnn::backward_filter()
{
    if (dataType == CUDNN_DATA_FLOAT)
    {
        timeBackwardWeights = rnn->backward_filter();
    }
    else if (dataType == CUDNN_DATA_HALF)
    {
        timeBackwardWeights = rnnfp16->backward_filter();
    }
}

void op_rnn_cudnn::backward(){}

std::string op_rnn_cudnn::get_fwd_algo_name() { return ""; }
std::string op_rnn_cudnn::get_bwd_data_name() { return ""; }
std::string op_rnn_cudnn::get_bwd_filter_name() { return ""; }

void op_rnn_cudnn::print_fwd_time(const float kernel_average_time)
{
    std::cout << std::left;
    std::cout << std::setw(46);
    std::cout << "GPU Kernel Time Forward RNN Elapsed: ";
    std::cout << std::setw(7);
    std::cout << std::setprecision(5);
    std::cout << timeForward << " ms";
    std::cout << std::endl;
}
void op_rnn_cudnn::print_bwd_time(const float kernel_average_time)
{
    std::cout << std::left;
    std::cout << std::setw(46);
    std::cout << "GPU Kernel Time Backward Data RNN Elapsed: ";
    std::cout << std::setw(7);
    std::cout << std::setprecision(5);
    std::cout << timeBackwardData << "ms";
    std::cout << std::endl;
}
void op_rnn_cudnn::print_wrw_time(const float kernel_average_time)
{
    std::cout << std::left;
    std::cout << std::setw(46);
    std::cout << "GPU Kernel Time Backward Weights RNN Elapsed: ";
    std::cout << std::setw(7);
    std::cout << std::setprecision(5);
    std::cout << timeBackwardWeights << "ms";
    std::cout << std::endl;
}

/*/// reference MIOpenDriver output
$ /opt/rocm-6.0.2/bin/MIOpenDriver rnn     -n 224 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -c 0 -F 0 -t 1 -w 1 -V 0
MIOpenDriver rnn -n 224 -W 224 -H 1000 -l 8 -b 1 -m lstm -p 0 -r 0 -k 32 -c 0 -F 0 -t 1 -w 1 -V 0
length of data sequence == 1 is short than time sequence == 32, padding the rest of data sequence with 224
length of data sequence == 1 is short than time sequence == 32, padding the rest of data sequence with 224
PRNG seed: 12345678
GPU Kernel Time Forward RNN Elapsed: 0.000000 ms
Wall-clock Time Forward RNN Elapsed: 4633.160645 ms
GPU Kernel Time Backward Data RNN Elapsed: 0.000000 ms
Wall-clock Time Backward Data RNN Elapsed: 161.234756 ms
GPU Kernel Time Backward Weights RNN Elapsed: 0.000000 ms
Wall-clock Time Backward Weights RNN Elapsed: 121.429344 ms
//*///
