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

template <class data_T, typename CONFIG_T> inline unsigned garnet_idx_from_real_val(data_T x) {
    if (x < 0)
        x = -x;

    // FIXME: infer data type conversion from training
    unsigned idx = (unsigned)((ap_fixed<32, 16>)x << CONFIG_T::exp_table_indexing_shmt);
    idx = (idx > CONFIG_T::exp_table_size - 1) ? CONFIG_T::exp_table_size - 1 : idx;
    idx = (idx < 0) ? 0 : idx;
    return idx;
}

template <class exp_table_T, typename CONFIG_T> void garnet_init_exp_table(exp_table_T table_out[CONFIG_T::exp_table_size]) {
    // Set exp for small distances to one to give room for optimizations
    table_out[0] = 1;
    // Set exp for large distances to zero to give room for optimizations
    table_out[CONFIG_T::exp_table_size - 1] = 0;

    for (unsigned i = 0; i < CONFIG_T::exp_table_size - 2; i++) {
        float val = (float)((ap_fixed<32, 16>)(i + 1) >> CONFIG_T::exp_table_indexing_shmt);
        exp_table_T exp_x = garnet_exp_fcn_float(-val * val);
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
            unsigned idx = garnet_idx_from_real_val<data_T, CONFIG_T>(distances[v * CONFIG_T::S + s]);
            weights[s * CONFIG_T::V + v] = exp_table[idx];
        }
    }
}

template <class res_T, typename CONFIG_T> res_T garnetlayer_acc_tree(res_T data[CONFIG_T::V]) {
    int D_tree = CONFIG_T::V_nbits + 1; // Include root node
    int W_tree = CONFIG_T::V;
    int w_current = W_tree;

    res_T acc_buf[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = acc_buf complete
#pragma HLS ARRAY_PARTITION variable = data complete

AccTreeDepth:
    for (int d = 0; d < D_tree; d++) {
#pragma HLS PIPELINE II = 1
    AccTreeWidth:
        for (int w = 0; w < W_tree; w++) {
#pragma HLS UNROLL
            if (d == 0) { // Leaf nodes
                acc_buf[w] = data[w];
            } else if (w < w_current) {
                acc_buf[w] = acc_buf[w * 2] + acc_buf[w * 2 + 1];
            }
        }
        // Tree width decreases on every layer
        w_current >>= 1;
    }

    // Root node is in first element and stores result
    return acc_buf[0];
}

template <class input1_T, class exp_table_T, class res_T, typename CONFIG_T>
void garnet_main_loop(input1_T input1[CONFIG_T::V * CONFIG_T::N], exp_table_T weights[CONFIG_T::V * CONFIG_T::S],
                      res_T res[CONFIG_T::S * CONFIG_T::N]) {
Aggregators:
    for (int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS PIPELINE II = 1
    Features:
        for (int n = 0; n < CONFIG_T::N; n++) {
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