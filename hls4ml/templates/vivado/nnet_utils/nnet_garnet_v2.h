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

template <class data_T, typename CONFIG_T>
inline ap_uint<CONFIG_T::exp_table_size_nbits> garnet_idx_from_real_val(data_T x) {
    if (x < 0)
        x = -x;

    // FIXME: infer data type conversion from training
    ap_uint<CONFIG_T::exp_table_size_nbits> max_idx = CONFIG_T::exp_table_size - 1;
    ap_fixed<x.width + CONFIG_T::exp_table_indexing_shmt, x.iwidth + CONFIG_T::exp_table_indexing_shmt> idx =
        ((ap_fixed<x.width + CONFIG_T::exp_table_indexing_shmt, x.iwidth + CONFIG_T::exp_table_indexing_shmt>)x
         << CONFIG_T::exp_table_indexing_shmt);
    if (idx > max_idx) {
        return max_idx;
    }
    return (ap_uint<CONFIG_T::exp_table_size_nbits>)idx;
}

template <class exp_table_T, typename CONFIG_T> void garnet_init_exp_table(exp_table_T table_out[CONFIG_T::exp_table_size]) {
    // Set exp for small distances to one to give room for optimizations
    table_out[0] = 1.0f / (float)CONFIG_T::V;
    // Set exp for large distances to zero to give room for optimizations
    table_out[CONFIG_T::exp_table_size - 1] = 0.0f;

    for (unsigned i = 1; i < CONFIG_T::exp_table_size - 1; i++) {
        float val = (float)((ap_fixed<32, 16>)(i + 1) >> CONFIG_T::exp_table_indexing_shmt);
        exp_table_T exp_x = garnet_exp_fcn_float(-val * val) / (float)CONFIG_T::V;
        table_out[i] = exp_x;
    }
}

template <class data_T, class exp_table_T, typename CONFIG_T>
void garnet_init_weights(data_T distances[CONFIG_T::V * CONFIG_T::S], exp_table_T exp_table[CONFIG_T::exp_table_size],
                         exp_table_T weights[CONFIG_T::V * CONFIG_T::S]) {
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
    int D_tree = CONFIG_T::V_nbits; // Include root node
    int W_tree = CONFIG_T::V;
    int w_current = W_tree / 2;

    res_T acc_buf[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = acc_buf complete
#pragma HLS ARRAY_PARTITION variable = data complete

InitAccBuffer:
    for (int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS UNROLL
        acc_buf[v] = data[v];
    }

AccTreeDepth:
    for (int d = 0; d < D_tree; d++) {
#pragma HLS PIPELINE II = 1
    AccTreeWidth:
        for (int w = 0; w < W_tree / 2; w++) {
#pragma HLS UNROLL
            if (w < w_current) {
                acc_buf[w] = acc_buf[w * 2] + acc_buf[w * 2 + 1];
            }
        }
        // Tree width decreases on every layer
        w_current >>= 1;
    }

    // Root node is in first element and stores result
    return acc_buf[0];
}

template <class input1_T, class input2_T, class exp_table_T, class res_T, typename CONFIG_T>
void garnet_main_loop(input1_T input1[CONFIG_T::V * CONFIG_T::N], input2_T input2[CONFIG_T::V * CONFIG_T::S],
                      exp_table_T exp_table[CONFIG_T::exp_table_size], res_T res[CONFIG_T::S * CONFIG_T::N]) {
Aggregators:
    for (int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS PIPELINE II = 1
        res_T feature_buf[CONFIG_T::V];
        res_T weight_buf[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = feature_buf complete
#pragma HLS ARRAY_PARTITION variable = weight_buf complete

    InitializeWeights:
        for (int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS UNROLL
            ap_uint<CONFIG_T::exp_table_size_nbits> idx =
                garnet_idx_from_real_val<input2_T, CONFIG_T>(input2[v * CONFIG_T::S + s]);
            exp_table_T w = exp_table[idx];
            weight_buf[v] = w;
        }
        res_T weighted_features = garnetlayer_acc_tree<res_T, CONFIG_T>(weight_buf);

    Features:
        for (int n = 0; n < CONFIG_T::N; n++) {
#pragma HLS UNROLL

        InitializeBuffers:
            for (int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS UNROLL
                feature_buf[v] = (res_T)(input1[v * CONFIG_T::N + n] * weight_buf[v]);
            }
            res_T h = garnetlayer_acc_tree<res_T, CONFIG_T>(feature_buf);
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
#pragma HLS ARRAY_PARTITION variable = exp_table complete

    garnet_init_exp_table<exp_table_T, CONFIG_T>(exp_table);
    garnet_main_loop<input1_T, input2_T, exp_table_T, res_T, CONFIG_T>(input1, input2, exp_table, res);
}

} // namespace nnet

#endif // NNET_GARNET_V2_H_