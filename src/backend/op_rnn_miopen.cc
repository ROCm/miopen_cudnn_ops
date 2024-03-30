#include "operator.hpp"
#include "backend.hpp"

op_rnn_miopen::op_rnn_miopen(void * desc): op_rnn(desc){}
op_rnn_miopen::~op_rnn_miopen() {}

void op_rnn_miopen::tune_op(){}
void op_rnn_miopen::forward(){}
void op_rnn_miopen::backward_data(){}
void op_rnn_miopen::backward_filter(){}
void op_rnn_miopen::backward(){}

std::string op_rnn_miopen::get_fwd_algo_name() { return ""; }
std::string op_rnn_miopen::get_bwd_data_name() { return ""; }
std::string op_rnn_miopen::get_bwd_filter_name() { return ""; }

void op_rnn_miopen::print_fwd_time(const float kernel_average_time) {}
void op_rnn_miopen::print_bwd_time(const float kernel_average_time) {}
void op_rnn_miopen::print_wrw_time(const float kernel_average_time) {}
