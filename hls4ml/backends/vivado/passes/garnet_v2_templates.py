import numpy as np

from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import GarNetLayer

garnetlayer_config_template = """struct config{index}: nnet::garnetlayer_config {{
static const unsigned V = {V};
static const unsigned V_nbits = {V_nbits};
static const unsigned S = {S};
static const unsigned N = {N};
static const unsigned exp_table_size = {exp_table_size};
static const unsigned exp_table_size_nbits = {exp_table_size_nbits};
static const unsigned exp_table_indexing_shmt = {exp_table_indexing_shmt};
typedef {exp_table_t} exp_table_t;
}};\n"""

garnetlayer_function_template = (
    'nnet::garnetlayer<{input1_t}, {input2_t}, {output_t}, {config}>({input1}, {input2}, {output});'
)

garnetlayer_include_list = ['nnet_utils/nnet_garnet_v2.h']


class GarNetLayerConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(GarNetLayer)
        self.template = garnetlayer_config_template

    def format(self, node):
        params = self._default_config_params(node)

        V = node.attributes['V']
        params['V'] = V  # Number of vertices (hits)
        params['V_nbits'] = int(np.ceil(np.log2(V)))
        params['S'] = node.attributes['S']  # Number of aggregators per vertex (hit)
        params['N'] = node.attributes['N']  # Number of encoded features per vertex

        # Calculate exponential table size and resolution from max input value during training
        max_dist_input = node.attributes['max_dist_input']

        # Make scale factor a power of two
        # This is the maximum representable value in our exp table
        scale_factor = 2 ** int(np.ceil(np.log2(max_dist_input)))

        # Keep scale factor at a sane level: exp(-4 * 4) is already very small (1.125e-7)
        # Otherwise, the exp table size might explode
        scale_factor = 4 if scale_factor > 4 else scale_factor

        # Here we want to make sure that the resolution remains for small values
        # exp(-1) = 0.37, but you might still get smaller values, which could then
        # directly get rendered to zero
        scale_factor = 2 if scale_factor < 2 else scale_factor

        resolution = 64 if node.attributes['exp_table_resolution'] is None else node.attributes['exp_table_resolution']
        resolution = 32 if scale_factor == 4 else resolution
        assert np.log2(scale_factor).is_integer(), "Scale factor must be a power of two"
        assert np.log2(resolution).is_integer(), "Exponential table resolution must be a power of two"

        exp_table_size = scale_factor * resolution
        params['exp_table_size'] = exp_table_size
        params['exp_table_size_nbits'] = int(np.log2(exp_table_size))
        params['exp_table_indexing_shmt'] = int(np.log2(resolution))

        # Two integer bits should be enough, since max(exp(-dist * dist)) = 1
        # Additionally keep one sign bit
        params['exp_table_t'] = 'ap_fixed<16,2>'

        return self.template.format(**params)


class GarNetLayerFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(GarNetLayer, include_header=garnetlayer_include_list)
        self.template = garnetlayer_function_template

    def format(self, node):
        assert len(node.inputs) == 2  # Encoded features and aggregator distances
        assert len(node.outputs) == 1

        params = self._default_function_params(node)

        input_encoded_features = node.get_input_variable(node.inputs[0])
        input_aggregated_distances = node.get_input_variable(node.inputs[1])
        output = node.get_output_variable()

        # Assign types
        params['input1_t'] = input_encoded_features.type.name
        params['input2_t'] = input_aggregated_distances.type.name
        params['output_t'] = output.type.name

        # Assign input and output data args
        params['input1'] = input_encoded_features.name
        params['input2'] = input_aggregated_distances.name
        params['output'] = output.name

        return self.template.format(**params)
