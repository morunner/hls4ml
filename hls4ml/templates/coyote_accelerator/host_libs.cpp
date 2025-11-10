#include "host_libs.hpp"
#include <chrono>
#include <cstddef>
#include <numeric>
#include <thread>
#include <tuple>

CoyoteInference::CoyoteInference(unsigned int batch_size, unsigned int in_size, unsigned int *out_sizes, unsigned int n_out)
    : batch_size(batch_size), in_size(in_size), coyote_thread(DEFAULT_VFPGA_ID, getpid()) {

    this->out_sizes.assign(out_sizes, out_sizes + n_out);

    for (unsigned int i = 0; i < batch_size; i++) {
        // Allocate memory using huge pages (HPF) for input and output tensors
        // We allow only one input because hls4ml flattens the input test data anyways,
        // so one needs to merge all input data into one .npy or .dat array (one column per sample)
        // With multiple outputs, the .data array can contain one feature on each column, hence
        // multiple outputs can be allowed

        // Inputs
        src_mems.emplace_back(
            (float *)coyote_thread.getMem({coyote::CoyoteAllocType::HPF, (uint)(in_size * sizeof(float))}));
        if (!src_mems[i]) {
            throw std::runtime_error("Could not allocate memory for inputs; exiting...");
        }

        // Create scatter-gather entry for this input
        coyote::localSg src_sg = {.addr = src_mems[i], .len = (uint)(in_size * sizeof(float)), .dest = 0};
        src_sgs.emplace_back(src_sg);

        // Outputs
        std::vector<float *> out_mems;
        std::vector<coyote::localSg> out_sgs;
        for (unsigned int j = 0; j < this->out_sizes.size(); j++) {
            unsigned int size = this->out_sizes[j];

            float *out_mem = (float *)coyote_thread.getMem({coyote::CoyoteAllocType::HPF, (uint)(size * sizeof(float))});
            if (!out_mem) {
                throw std::runtime_error("Could not allocate memory for outputs; exiting...");
            }

            // Increment .dest corresponding to a new axi stream for each tensor
            coyote::localSg sg = {.addr = out_mem, .len = (uint)(size * sizeof(float)), .dest = j};
            out_mems.emplace_back(out_mem);
            out_sgs.emplace_back(sg);
        }
        dst_mems.emplace_back(std::move(out_mems));
        dst_sgs.emplace_back(std::move(out_sgs));
    }
}

CoyoteInference::~CoyoteInference() {}

void CoyoteInference::flush() {
    // Reset output tensors to zero
    for (unsigned int i = 0; i < batch_size; i++) {
        for (unsigned int j = 0; j < out_sizes.size(); j++) {
            memset(dst_mems[i][j], 0, out_sizes[j]);
        }
    }

    // Clear completion counters
    coyote_thread.clearCompleted();
}

void CoyoteInference::predict() {
    // Coyote uses the so-called invoke function to run operation in vFPGAs.
    // In this case, the operation is LOCAL_TRANSFER, and the flow of data is:
    // host memory (input data) => vFPGA (hls4ml model) => host memory (output
    // data)
    for (unsigned int i = 0; i < batch_size; i++) {
        coyote_thread.invoke(coyote::CoyoteOper::LOCAL_READ, src_sgs[i]);
        for (const auto &dst_sg : dst_sgs[i]) {
            coyote_thread.invoke(coyote::CoyoteOper::LOCAL_WRITE, dst_sg);
        }
    }

    // Poll on completion; each batch, input and output increment the counter by
    // one
    while (coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_WRITE) != out_sizes.size() * batch_size ||
           coyote_thread.checkCompleted(coyote::CoyoteOper::LOCAL_READ) != batch_size) {
    }
}

void CoyoteInference::set_data(float *x, unsigned int i) {
    // Simply copy from one buffer to the other
    for (int j = 0; j < in_size; j++) {
        src_mems[i][j] = x[j];
    }
}

void CoyoteInference::get_predictions(float *predictions, unsigned int i) {
    const unsigned int n_out = out_sizes.size();
    unsigned int offset = 0;
    for (unsigned int j = 0; j < n_out; j++) {
        for (unsigned int k = 0; k < out_sizes[j]; k++) {
            predictions[offset + k] = dst_mems[i][j][k];
        }
        offset += out_sizes[j];
    }
}

// C API for the CoyoteInference class; so that it can be used from Python or
// other languages Better option would be to use something like pybind11, but
// the implementation is simple enough for now.
extern "C" {
CoyoteInference *init_model_inference(unsigned int batch_size, unsigned int in_size, unsigned int *out_sizes,
                                      unsigned int n_out) {
    return new CoyoteInference(batch_size, in_size, out_sizes, n_out);
}

void free_model_inference(CoyoteInference *obj) { delete obj; }

void flush(CoyoteInference *obj) { obj->flush(); }

void predict(CoyoteInference *obj) { obj->predict(); }

void set_inference_data(CoyoteInference *obj, float *x, unsigned int i) { obj->set_data(x, i); }

void get_inference_predictions(CoyoteInference *obj, float *predictions, unsigned int i) {
    obj->get_predictions(predictions, i);
}
}
