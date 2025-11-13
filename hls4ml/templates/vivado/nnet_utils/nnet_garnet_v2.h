#ifndef NNET_GARNET_V2_H_
#define NNET_GARNET_V2_H_

#include "hls_math.h"
#include "nnet_common.h"

namespace nnet {

struct garnetlayer_config {
    static const unsigned V = 128;
    static const unsigned S = 8;
    static const unsigned N = 16;
};

inline float garnet_exp_fcn_float(float input) { return std::exp(input); }

template <class data_T, typename CONFIG_T> ap_uint<CONFIG_T::exp_table_size_nbits> garnet_idx_from_real_val(data_T x) {
    if (x < 0)
        x = -x;

    // Instead of simply shifting by CONFIG_T::exptable_indexing_shmt, we slice off the required bits for indexing.
    unsigned int cutoff = (x.width - x.iwidth) + (CONFIG_T::exp_table_size_nbits - CONFIG_T::exp_table_indexing_shmt);

    // All integer bits which form a number > CONFIG_T::exp_table_size - 1 (> cutoff) when shifted can directly be discarded.
    if (x.range(x.width - 1, cutoff) != 0) {
        return (ap_uint<CONFIG_T::exp_table_size_nbits>)CONFIG_T::exp_table_size - 1;
    }

    // Slice the bits such that the op correspond to a `<< CONFIG_T::exp_table_indexing_shmt` operation
    return (ap_uint<CONFIG_T::exp_table_size_nbits>)x(cutoff, cutoff - CONFIG_T::exp_table_size_nbits);
}

template <class exp_table_T, typename CONFIG_T> void garnet_init_exp_table(exp_table_T table_out[CONFIG_T::exp_table_size]) {
#pragma HLS ARRAY_PARTITION variable = table_out complete
    // Set exp for small distances to one to give room for optimizations
    table_out[0] = 1;
    // Set exp for large distances to zero to give room for optimizations
    table_out[CONFIG_T::exp_table_size - 1] = 0;

InitExpTable:
    for (unsigned i = 0; i < CONFIG_T::exp_table_size - 2; i++) {
#pragma HLS UNROLL
        // Max int amount is CONFIG_T::exp_table_size - 1, max fraction is 0 >> CONFIG_T::exp_table_indexing_shmt
        float val = (float)((ap_ufixed<CONFIG_T::exp_table_size_nbits + CONFIG_T::exp_table_indexing_shmt,
                                       CONFIG_T::exp_table_size_nbits>)(i + 1) >>
                            CONFIG_T::exp_table_indexing_shmt);
        exp_table_T exp_x = garnet_exp_fcn_float(-val * val);
        table_out[i] = exp_x;
    }
}

template <class data_T, class exp_table_T, typename CONFIG_T>
void garnet_init_weights(data_T distances[CONFIG_T::V * CONFIG_T::S], exp_table_T exp_table[CONFIG_T::exp_table_size],
                         exp_table_T weights[CONFIG_T::V * CONFIG_T::S]) {
#pragma HLS ARRAY_PARTITION variable = distances complete
#pragma HLS ARRAY_PARTITION variable = exp_table complete
#pragma HLS ARRAY_PARTITION variable = weights complete

InitWeightsOuter:
    for (int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
    InitWeightsInner:
        for (int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS UNROLL
            ap_uint<CONFIG_T::exp_table_size_nbits> idx =
                garnet_idx_from_real_val<data_T, CONFIG_T>(distances[v * CONFIG_T::S + s]);
            weights[s * CONFIG_T::V + v] = exp_table[idx];
        }
    }
}

template <class res_T, typename CONFIG_T> res_T garnetlayer_acc_tree(res_T data[CONFIG_T::V]) {
    int D_tree = CONFIG_T::V_nbits + 1; // Include root node
    int W_tree = CONFIG_T::V;
    int w_current = W_tree;

    res_T acc_buf[CONFIG_T::V];
    res_T acc_buf_next[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = acc_buf complete
#pragma HLS ARRAY_PARTITION variable = acc_buf_next complete
#pragma HLS ARRAY_PARTITION variable = data complete

AccTreeDepth:
    for (int d = 0; d < D_tree; d++) {
#pragma HLS UNROLL
    AccTreeWidth:
        for (int w = 0; w < W_tree; w++) {
#pragma HLS UNROLL
            if (d == 0) { // Leaf nodes
                acc_buf[w] = data[w];
            } else if (w < w_current) {
                acc_buf[w] = acc_buf_next[w * 2] + acc_buf_next[w * 2 + 1];
            } else { // Set unused array elements to zero to allow potential HLS optimizations
                acc_buf[w] = 0;
            }
            // Shift register
            acc_buf_next[w] = acc_buf[w];
        }
        // Tree width decreases on every layer
        w_current >>= 1;
    }

    // Root node is in first element and stores result
    return acc_buf_next[0];
}

template <class input1_T, class exp_table_T, class res_T, typename CONFIG_T>
void garnet_main_loop(input1_T input1[CONFIG_T::V * CONFIG_T::N], exp_table_T weights[CONFIG_T::V * CONFIG_T::S],
                      res_T res[CONFIG_T::S * CONFIG_T::N]) {
Features:
    for (int n = 0; n < CONFIG_T::N; n++) {
#pragma HLS PIPELINE II = 1
    Aggregators:
        for (int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
            res_T feature_buf[CONFIG_T::V];
            res_T weight_buf[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = feature_buf complete
#pragma HLS ARRAY_PARTITION variable = weight_buf complete

        InitializeBuffers:
            for (int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS UNROLL
                feature_buf[v] = (res_T)(input1[v * CONFIG_T::N + n] * weights[s * CONFIG_T::V + v] >> CONFIG_T::V_nbits);
                weight_buf[v] = (res_T)(weights[s * CONFIG_T::V + v] >> CONFIG_T::V_nbits);
            }

            res_T h = garnetlayer_acc_tree<res_T, CONFIG_T>(feature_buf);
            res_T weighted_features = garnetlayer_acc_tree<res_T, CONFIG_T>(weight_buf);
            res[s * CONFIG_T::N + n] = h * weighted_features;
        }
    }
}

template <class input1_T, class input2_T, class res_T, class exp_table_T, typename CONFIG_T>
void garnetlayer(input1_T input1[CONFIG_T::V * CONFIG_T::N], input2_T input2[CONFIG_T::V * CONFIG_T::S],
                 res_T res[CONFIG_T::S * CONFIG_T::N]) {
#pragma HLS ARRAY_PARTITION variable = input1 complete
#pragma HLS ARRAY_PARTITION variable = input2 complete
#pragma HLS ARRAY_PARTITION variable = res complete

    exp_table_T exp_table[CONFIG_T::exp_table_size];
    exp_table_T weights[CONFIG_T::V * CONFIG_T::S];
#pragma HLS ARRAY_PARTITION variable = exp_table complete
#pragma HLS ARRAY_PARTITION variable = weights complete

#pragma HLS PIPELINE II = 16
    garnet_init_exp_table<exp_table_T, CONFIG_T>(exp_table);
    garnet_init_weights<input2_T, exp_table_T, CONFIG_T>(input2, exp_table, weights);
    garnet_main_loop<input1_T, exp_table_T, res_T, CONFIG_T>(input1, weights, res);
}

} // namespace nnet

#endif // NNET_GARNET_V2_H_