#ifndef NNET_AXI_UTILS_H_
#define NNET_AXI_UTILS_H_

#include "ap_axi_sdata.h"

namespace nnet {

// Converts an array of data (fixed-point numbers) into 512-bit AXI stream packets; see model_wrapper.hpp for usage
template <class array_T, class axi_T, unsigned int SIZE, unsigned int AXI_BITS, unsigned int PRECISION>
void data_to_axi_stream(array_T data_in[SIZE], hls::stream<ap_axiu<AXI_BITS, 0, 0, 0>> &axi_out) {
#pragma HLS INLINE OFF
#pragma HLS PIPELINE

    constexpr const unsigned int ELEMENTS_PER_AXI = AXI_BITS / PRECISION;
    constexpr const unsigned int NUM_BEATS = (SIZE + ELEMENTS_PER_AXI - 1) / ELEMENTS_PER_AXI;

    for (unsigned int i = 0; i < NUM_BEATS; i++) {
        if (i == NUM_BEATS - 1) {
            ap_axiu<AXI_BITS, 0, 0, 0> axi_packet;
            unsigned int index = i * ELEMENTS_PER_AXI;

            for (unsigned int j = 0; j < SIZE - index; j++) {
#pragma HLS UNROLL

                axi_T axi_tmp = axi_T(data_in[index + j]);
                ap_uint<PRECISION> axi_bits = *reinterpret_cast<ap_uint<PRECISION> *>(&axi_tmp);
                axi_packet.data.range((j + 1) * PRECISION - 1, j * PRECISION) = axi_bits;
            }

            axi_packet.last = 1;
            axi_out.write(axi_packet);

        } else {
            ap_axiu<AXI_BITS, 0, 0, 0> axi_packet;
            unsigned int index = i * ELEMENTS_PER_AXI;

            for (unsigned int j = 0; j < ELEMENTS_PER_AXI; j++) {
#pragma HLS UNROLL

                axi_T axi_tmp = axi_T(data_in[index + j]);
                ap_uint<PRECISION> axi_bits = *reinterpret_cast<ap_uint<PRECISION> *>(&axi_tmp);
                axi_packet.data.range((j + 1) * PRECISION - 1, j * PRECISION) = axi_bits;
            }

            axi_packet.last = 0;
            axi_out.write(axi_packet);
        }
    }
}

// Unpacks beats of 512-bit AXI beats into an array of data (fixed-point numbers) see model_wrapper.hpp for usage
template <class array_T, class axi_T, unsigned int SIZE, unsigned int AXI_BITS, unsigned int PRECISION>
void axi_stream_to_data(hls::stream<ap_axiu<AXI_BITS, 0, 0, 0>> &axi_in, array_T data_out[SIZE]) {
#pragma HLS INLINE off

    static_assert(PRECISION == 32, "Currently only a PRECISION of 32 bits is supported for conversion to fixed point");

    constexpr const unsigned int ELEMENTS_PER_AXI = AXI_BITS / PRECISION;
    constexpr const unsigned int NUM_BEATS = (SIZE + ELEMENTS_PER_AXI - 1) / ELEMENTS_PER_AXI;

    array_T data_buffer[SIZE];
#pragma HLS ARRAY_PARTITION variable = data_buffer complete

    for (unsigned int i = 0; i < NUM_BEATS; i++) {
#pragma HLS PIPELINE II = 1
        ap_axiu<AXI_BITS, 0, 0, 0> axi_packet = axi_in.read();

        unsigned int index = i * ELEMENTS_PER_AXI;

        for (unsigned int j = 0; j < ELEMENTS_PER_AXI; j++) {
#pragma HLS UNROLL
#pragma HLS DEPENDENCE variable = data_out inter false

            if (index + j < SIZE) {
                ap_uint<PRECISION> axi_bits = axi_packet.data.range((j + 1) * PRECISION - 1, j * PRECISION);

                union {
                    unsigned int axi_bits_as_uint;
                    float axi_bits_asfloat;
                } converter;
                converter.axi_bits_as_uint = axi_bits.to_uint();
                float data_float = converter.axi_bits_asfloat;

                array_T data = data_float;
                data_buffer[index + j] = data;
            }
        }
    }

    for (unsigned int i = 0; i < SIZE; i++) {
#pragma HLS UNROLL
        data_out[i] = data_buffer[i];
    }
}

} // namespace nnet

#endif