#ifndef NNET_GARNET_V2_H_
#define NNET_GARNET_V2_H_

#include "hls_math.h"
#include "nnet_common.h"

#define LOG2V 7

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
    unsigned idx = (unsigned)((ap_fixed<32, 16>)x << CONFIG_T::exp_table_shmt);
    idx = (idx > CONFIG_T::exp_table_size - 1) ? CONFIG_T::exp_table_size - 1 : idx;
    idx = (idx < 0) ? 0 : idx;
    return idx;
}

template <class data_T, typename CONFIG_T>
void garnet_init_exp_table(typename CONFIG_T::exp_table_t table_out[CONFIG_T::exp_table_size]) {
    // Set exp for small distances to one to give room for optimizations
    table_out[0] = 1;
    // Set exp for large distances to zero to give room for optimizations
    table_out[CONFIG_T::exp_table_size - 1] = 0;

    for (unsigned i = 0; i < CONFIG_T::exp_table_size - 2; i++) {
        // FIXME: infer data type conversion from training
        float val = (float)((ap_fixed<32, 16>)(i + 1) >> CONFIG_T::exp_table_shmt);
        typename CONFIG_T::exp_table_t exp_x = garnet_exp_fcn_float(-val * val);
        table_out[i] = exp_x;
    }
}

template <class data_T, typename CONFIG_T>
void garnet_init(data_T distances[CONFIG_T::V * CONFIG_T::S],
                 typename CONFIG_T::exp_table_t weights[CONFIG_T::V * CONFIG_T::S]) {
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
#pragma HLS ARRAY_PARTITION variable = exp_table complete
    garnet_init_exp_table<data_T, CONFIG_T>(exp_table);

InitWeightsOuter:
    for (int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS PIPELINE II = 1
    InitWeightsInner:
        for (int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS UNROLL
            unsigned idx = garnet_idx_from_real_val<data_T, CONFIG_T>(distances[v * CONFIG_T::S + s]);
            weights[s * CONFIG_T::V + v] = exp_table[idx];
        }
    }
}

template <class res_T, typename CONFIG_T> res_T garnetlayer_acc_tree(res_T data[CONFIG_T::V]) {
    int D_tree = LOG2V + 1; // Include root node
    int W_tree = CONFIG_T::V;
    int w_current = W_tree;

    res_T mac_buf[CONFIG_T::V];
    res_T mac_buf_next[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = mac_buf off = true
#pragma HLS ARRAY_PARTITION variable = mac_buf_next off = true

    for (int d = 0; d < D_tree; d++) {
#pragma HLS PIPELINE II = 1
        for (int w = 0; w < W_tree; w++) {
#pragma HLS UNROLL
            if (d == 0) { // Leaf nodes
                mac_buf[w] = data[w];
            } else if (w < w_current) {
                mac_buf[w] = mac_buf_next[w * 2] + mac_buf_next[w * 2 + 1];
            } else { // Set unused array elements to zero to allow potential HLS optimizations
                mac_buf[w] = 0;
            }
            // Shift register
            mac_buf_next[w] = mac_buf[w];
        }
        // Tree width decreases on every layer
        w_current >>= 1;
    }

    // Root node is in first element and stores result
    return mac_buf_next[0];
}

template <class input1_T, class input2_T, class res_T, typename CONFIG_T>
void garnetlayer(input1_T input1[CONFIG_T::V * CONFIG_T::N], input2_T input2[CONFIG_T::V * CONFIG_T::S],
                 res_T res[CONFIG_T::S * CONFIG_T::N]) {
#pragma HLS ARRAY_PARTITION variable = input1 complete
#pragma HLS ARRAY_PARTITION variable = input2 complete
#pragma HLS ARRAY_PARTITION variable = res complete

    typename CONFIG_T::exp_table_t weights[CONFIG_T::V * CONFIG_T::S];
#pragma HLS ARRAY_PARTITION variable = weights complete

    garnet_init<input2_T, CONFIG_T>(input2, weights);

Aggregators:
    for (int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS PIPELINE II = 1
        res_T h[CONFIG_T::N];
#pragma HLS ARRAY_PARTITION variable = h complete

    Features:
        for (int n = 0; n < CONFIG_T::N; n++) {
#pragma HLS UNROLL
            res_T feature_buf[CONFIG_T::V];
            res_T weight_buf[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = feature_buf complete
#pragma HLS ARRAY_PARTITION variable = weight_buf complete

            for (int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS UNROLL
                feature_buf[v] = (res_T)(input1[v * CONFIG_T::N + n] * weights[s * CONFIG_T::V + v] >> CONFIG_T::V_nbits);
                weight_buf[v] = (res_T)(weights[s * CONFIG_T::V + v] >> CONFIG_T::V_nbits);
            }

            h[n] = garnetlayer_acc_tree<res_T, CONFIG_T>(feature_buf);
            res[s * CONFIG_T::N + n] = h[n] * garnetlayer_acc_tree<res_T, CONFIG_T>(weight_buf);
        }
    }
}

} // namespace nnet

#endif // NNET_GARNET_V2_H_