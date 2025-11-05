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
}};\n"""

garnetlayer_function_template = (
    'nnet::garnetlayer<{input1_t}, {input2_t}, {output_t}, {exp_table_t}, {config}>({input1}, {input2}, {output});'
)

garnetlayer_include_list = ['nnet_utils/nnet_garnet_v2.h']


class GarNetLayerConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(GarNetLayer)
        self.template = garnetlayer_config_template

    def format(self, node):
        params = self._default_config_params(node)

        V = node.get_attr('V')
        params['V'] = V  # Number of vertices (hits)
        params['V_nbits'] = int(np.ceil(np.log2(V)))
        params['S'] = node.get_attr('S')  # Number of aggregators per vertex (hit)
        params['N'] = node.get_attr('N')  # Number of encoded features per vertex

        # Calculate exponential table size and resolution from max input value during training
        exp_table_attr = node.get_attr('exponential_table')
        scale_factor = exp_table_attr['ScaleFactor']
        resolution = exp_table_attr['Resolution']
        exp_table_size = scale_factor * resolution

        params['exp_table_size'] = exp_table_size
        params['exp_table_size_nbits'] = int(np.log2(exp_table_size))
        params['exp_table_indexing_shmt'] = int(np.log2(resolution))

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
        exp_table = node.get_attr('exp_table_t')

        # Assign types
        params['input1_t'] = input_encoded_features.type.name
        params['input2_t'] = input_aggregated_distances.type.name
        params['output_t'] = output.type.name
        params['exp_table_t'] = exp_table.name

        # Assign input and output data args
        params['input1'] = input_encoded_features.name
        params['input2'] = input_aggregated_distances.name
        params['output'] = output.name
        params['exp_table'] = 'exp_table'

        return self.template.format(**params)
