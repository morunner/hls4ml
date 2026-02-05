#ifndef NNET_AXI_UTILS_STREAM_H
#define NNET_AXI_UTILS_STREAM_H

#include "ap_axi_sdata.h"

namespace nnet {

// Converts a stream of data (fixed-point numbers) into 512-bit AXI stream packets; see model_wrapper.hpp for usage
template <class array_T, class axi_T, unsigned int SIZE, unsigned int AXI_BITS, unsigned int PRECISION>
void data_to_axi_stream(hls::stream<array_T> &data_in, hls::stream<ap_axiu<AXI_BITS, 0, 0, 0>> &axi_out) {
    #pragma HLS INLINE OFF

    constexpr const unsigned int ELEMENTS_PER_AXI = (SIZE <= (AXI_BITS / PRECISION)) ? SIZE : (AXI_BITS / PRECISION);

    unsigned int index = 0;
    ap_axiu<AXI_BITS, 0, 0, 0> axi_packet;
    axi_packet.keep = -1;

    for (int i = 0; i < SIZE / array_T::size; i++) {
        #pragma HLS PIPELINE II = 1

        array_T in_data = data_in.read();

        for (int j = 0; j < array_T::size; j++) {
            #pragma HLS UNROLL

            axi_T axi_tmp = (axi_T)in_data[j];

            union {
                axi_T f;
                unsigned int i;
            } u;
            u.f = axi_tmp;
            ap_uint<PRECISION> axi_bits = u.i;

            axi_packet.data.range((index + 1) * PRECISION - 1, index * PRECISION) = axi_bits;
            index++;

            if (index == ELEMENTS_PER_AXI) {
                bool is_last = (i == (SIZE / array_T::size) - 1) && (j == array_T::size - 1);
                axi_packet.last = is_last ? 1 : 0;

                axi_out.write(axi_packet);
                index = 0;
            }
        }
    }

    if (index != 0) {
        axi_packet.last = 1;
        axi_out.write(axi_packet);
    }
}

// Unpacks beats of 512-bit AXI beats into a stream of data (fixed-point numbers)
template <class array_T, class axi_T, unsigned int SIZE, unsigned int AXI_BITS, unsigned int PRECISION>
void axi_stream_to_data(hls::stream<ap_axiu<AXI_BITS, 0, 0, 0>> &axi_in, hls::stream<array_T> &data_out) {
    #pragma HLS INLINE off

    static_assert(PRECISION == 32, "Currently only a PRECISION of 32 bits is supported for conversion");

    constexpr const unsigned int ELEMENTS_PER_AXI = AXI_BITS / PRECISION;
    constexpr const unsigned int NUM_BEATS = (SIZE + ELEMENTS_PER_AXI - 1) / ELEMENTS_PER_AXI;
    constexpr const unsigned int OUTPUT_PACK_SIZE = array_T::size;

    array_T tmp_pack;
    #pragma HLS DATA_PACK variable = tmp_pack

    unsigned int pack_idx = 0;

    for (unsigned int i = 0; i < NUM_BEATS; i++) {
        #pragma HLS PIPELINE II = 1
        ap_axiu<AXI_BITS, 0, 0, 0> axi_packet = axi_in.read();

        unsigned int index = i * ELEMENTS_PER_AXI;

        for (unsigned int j = 0; j < ELEMENTS_PER_AXI; j++) {
            #pragma HLS UNROLL
            if (index + j < SIZE) {
                ap_uint<PRECISION> axi_bits = axi_packet.data.range((j + 1) * PRECISION - 1, j * PRECISION);

                union {
                    unsigned int axi_bits_as_uint;
                    float axi_bits_asfloat;
                } converter;

                converter.axi_bits_as_uint = axi_bits.to_uint();
                float data_float = converter.axi_bits_asfloat;

                typename array_T::value_type fixed_val = (typename array_T::value_type)data_float;

                tmp_pack[pack_idx] = fixed_val;
                pack_idx++;

                if (pack_idx == OUTPUT_PACK_SIZE) {
                    data_out.write(tmp_pack);
                    pack_idx = 0;
                }
            }
        }
    }
}

} // namespace nnet

#endif
