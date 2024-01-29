import os
import unittest

import mlx.core as mx
import numpy as np
import onnx.backend.test

from mlx.onnx import MlxBackend, MlxBackendWrapper


# need to conver to numpy for the testing suite
class TestMlxBackend(MlxBackend):
    def __init__(self, model):
        super().__init__(model)

    def run(self, inputs, **kwargs):
        t = super().run(inputs, **kwargs)
        return tuple(
            np.array(x) if isinstance(x, mx.array) else [np.array(i) for i in x]
            for x in t
        )


class TestMlxBackendWrapper(MlxBackendWrapper):
    @classmethod
    def prepare(cls, model: onnx.ModelProto, device: str):
        return TestMlxBackend(model)


btest = onnx.backend.test.BackendTest(TestMlxBackendWrapper, __name__)

# btest.include("test_sce_*")
btest.exclude("test_sce_*")
# TODO: these are upcasting to float32
btest.exclude("test_div_uint8_cpu")

# TODO: Debug these errors
btest.exclude("test_onehot_negative_indices_cpu")
# TODO: Implement
btest.exclude("test_ReplicationPad2d_*")
btest.exclude("test_wrap_pad_*")
btest.exclude("test_ReflectionPad2d_*")
btest.exclude("test_edge_*")
btest.exclude("test_reflect_pad_cpu")
btest.exclude("test_center_crop_pad_*")
btest.exclude("test_operator_pad_*")

btest.exclude("test_operator_convtranspose_cpu")
btest.exclude("test_ConvTranspose2d_*")
btest.exclude("test_ConstantPad2d_*")
btest.exclude("test_convtranspose_*")

# TODO: Implement dilations / col format
btest.exclude("test_averagepool_2d_dilations_cpu")
btest.exclude("test_averagepool_3d_dilations_*")
btest.exclude("test_maxpool_with_argmax_2d_precomputed_pads_cpu")
btest.exclude("test_maxpool_2d_dilations_cpu")
btest.exclude("test_maxpool_with_argmax_2d_precomputed_strides_cpu")
btest.exclude("test_maxpool_3d_dilations_*")
btest.exclude("test_MaxPool1d_stride_padding_dilation_cpu")
btest.exclude("test_MaxPool2d_stride_padding_dilation_cpu")
btest.exclude("test_Conv2d_groups_thnn_cpu")

btest.exclude("test_maxunpool_*")

# TODO: These are training parameters
btest.exclude("test_batchnorm_example_training_mode_cpu")
btest.exclude("test_batchnorm_epsilon_training_mode_cpu")
btest.exclude("test_BatchNorm*")

btest.exclude("test_gelu_tanh_*")
btest.exclude("test_bitshift_*")
btest.exclude("test_bitwise_*")
btest.exclude("test_gathernd_*")
btest.exclude("test_tfidfvectorizer_*")
btest.exclude("test_unique_*")
btest.exclude("test_einsum_*")
btest.exclude("test_convinteger_*")
btest.exclude("test_nonmaxsuppression_*")
btest.exclude("test_hardmax_*")
btest.exclude("test_scatternd_*")
btest.exclude("test_scatter_*")
btest.exclude("test_scatter_elements_*")
btest.exclude("test_gridsample_*")
btest.exclude("test_bernoulli_*")

btest.exclude("test_roialign_*")
btest.exclude("test_nonzero_example_cpu")
btest.exclude("test_upsample_nearest_cpu")
btest.exclude("test_lppool_*")
btest.exclude("test_reversesequence_*")
btest.exclude("test_col2im_*")
btest.exclude("test_deform_conv_*")
btest.exclude("test_basic_deform_conv_*")
btest.exclude("test_stft_*")
btest.exclude("test_det_*")
btest.exclude("test_dft_*")
btest.exclude("test_adagrad_*")
btest.exclude("test_momentum_*")
btest.exclude("test_nesterov_momentum_cpu")
btest.exclude("test_adam_*")

btest.exclude("test_gru_*")
btest.exclude("test_rnn_*")
btest.exclude("test_simple_rnn_*")
btest.exclude("test_lstm_*")

btest.exclude("test_training_dropout_*")

btest.exclude("test_melweightmatrix_cpu")
btest.exclude("test_resize_*")
btest.exclude("test_regex_*")

btest.exclude("test_nllloss_*")
btest.exclude("test_mvn_*")

btest.exclude("test_ai_onnx_ml_*")

# TODO: Quantize ops
btest.exclude("test_qlinearconv_*")
btest.exclude("test_qlinearmatmul_*")
btest.exclude("test_quantizelinear_*")
btest.exclude("test_dynamicquantizelinear_*")
btest.exclude("test_dequantizelinear_*")

# Exclude conv due to either dilation or groups
btest.exclude("test_Conv1d_dilated_cpu")
btest.exclude("test_Conv1d_groups_cpu")
btest.exclude("test_Conv2d_depthwise_cpu")
btest.exclude("test_Conv2d_depthwise_padded_cpu")
btest.exclude("test_Conv2d_depthwise_strided_cpu")
btest.exclude("test_Conv2d_depthwise_with_multiplier_cpu")
btest.exclude("test_Conv2d_dilated_cpu")
btest.exclude("test_Conv2d_groups_cpu")
btest.exclude("test_Conv3d_*")
btest.exclude("test_bvlc_alexnet_cpu")
btest.exclude("test_squeezenet_cpu")
btest.exclude("test_shufflenet_cpu")

btest.exclude("test_cast_no_saturate_FLOAT_to_FLOAT8*")
btest.exclude("test_cast_FLOAT_to_FLOAT8*")
btest.exclude("test_cast_no_saturate_FLOAT16_to_FLOAT8*")
btest.exclude("test_cast_FLOAT16_to_FLOAT8*")
btest.exclude("test_cast_FLOAT_to_BFLOAT16_cpu")
btest.exclude("test_cast_STRING_to_FLOAT_cpu")
btest.exclude("test_cast_BFLOAT16_to_FLOAT_cpu")
btest.exclude("test_cast_FLOAT_to_STRING_cpu")

btest.exclude("test_castlike_FLOAT_to_BFLOAT16*")
btest.exclude("test_castlike_FLOAT_to_STRING*")
btest.exclude("test_castlike_BFLOAT16_*")
btest.exclude("test_castlike_STRING*")
btest.exclude("test_castlike_FLOAT_to_FLOAT8*")

# TODO: need to go through and handle these better
btest.exclude("test_argmax_keepdims_example_select_last_index_cpu")
btest.exclude("test_argmax_negative_axis_keepdims_example_select_last_index_cpu")
btest.exclude("test_argmax_no_keepdims_example_select_last_index_cpu")
btest.exclude("test_argmin_no_keepdims_example_select_last_index_cpu")
btest.exclude("test_argmin_negative_axis_keepdims_example_select_last_index_cpu")
btest.exclude("test_argmin_keepdims_example_select_last_index_cpu")
btest.exclude("test_scan_sum_cpu")


# TODO: Graph tests
btest.exclude("test_range_float_type_positive_delta_expanded_cpu")
btest.exclude("test_range_int32_type_negative_delta_expanded_cpu")
btest.exclude("test_scan9_sum_cpu")
btest.exclude("test_loop16_seq_none_cpu")
btest.exclude("test_loop13_seq_cpu")
btest.exclude("test_loop11_cpu")
btest.exclude("test_if_*")
btest.exclude("test_affine_grid_*")

# TODO: Add gradient support
btest.exclude("test_gradient_*")

# TODO: Investigate
btest.exclude("test_mod_mixed_sign_int8_cpu")
btest.exclude("test_mod_mixed_sign_int64_cpu")
btest.exclude("test_mod_mixed_sign_int32_cpu")
btest.exclude("test_mod_mixed_sign_int16_cpu")

btest.exclude("test_sequence_map_*")
btest.exclude("test_strnorm_*")
btest.exclude("string")

# float64 datatype
btest.exclude("test_castlike_FLOAT_to_DOUBLE*")
btest.exclude("test_castlike_FLOAT16_to_DOUBLE*")
btest.exclude("test_sequence_model7_cpu")
btest.exclude("test_max_float64_cpu")
btest.exclude("test_min_float64_cpu")
btest.exclude("test_reduce_log_sum_exp_*")
btest.exclude("test_operator_addconstant_cpu")
btest.exclude("test_operator_add_size1_singleton_broadcast_cpu")
btest.exclude("test_operator_add_broadcast_cpu")
btest.exclude("test_operator_add_size1_broadcast_cpu")
btest.exclude("test_operator_add_size1_right_broadcast_cpu")
btest.exclude("test_cumsum_*")
btest.exclude("test_eyelike_with_dtype_cpu")
btest.exclude("test_mod_mixed_sign_float64_cpu")
btest.exclude("test_cast_FLOAT16_to_DOUBLE_cpu")
btest.exclude("test_cast_FLOAT_to_DOUBLE_cpu")

if os.getenv("MODELS", "0") == "0":
    for x in btest.test_suite:
        if "OnnxBackendRealModelTest" in str(type(x)):
            btest.exclude(str(x).split(" ")[0])

globals().update(btest.enable_report().test_cases)

if __name__ == "__main__":
    unittest.main()
