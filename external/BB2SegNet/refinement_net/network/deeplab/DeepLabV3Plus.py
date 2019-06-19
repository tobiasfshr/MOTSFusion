from external.BB2SegNet.refinement_net.network.Layer import Layer
from . import model, common
from external.BB2SegNet.refinement_net.datasets.Dataset import unnormalize


class DeepLabV3Plus(Layer):
  def __init__(self, name, inputs, n_features, tower_setup, l2=Layer.L2_DEFAULT):
    super().__init__()
    assert len(inputs) == 1, len(inputs)
    inputs = inputs[0]
    # revert normalization since DeepLab will do it's own normalization
    # also multiply by 255 since DeepLab expects values in the range 0..255
    inputs = unnormalize(inputs) * 255
    if tower_setup.is_training:
      crop_size = [int(inputs.get_shape()[1]), int(inputs.get_shape()[2])]
    else:
      crop_size = None
    # for choosing options and a pretrained model to load, see here: https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
    model_options = common.ModelOptions(
      outputs_to_num_classes={"features": n_features},
      crop_size=crop_size,
      atrous_rates=[6, 12, 18],
      output_stride=16,
      merge_method="max",
      add_image_level_feature=True,
      aspp_with_batch_norm=True,
      aspp_with_separable_conv=True,
      multi_grid=None,
      decoder_output_stride=4,
      decoder_use_separable_conv=True,
      logits_kernel_size=1,
      model_variant="xception_65")
    outputs_to_scales_to_logits = model.multi_scale_logits(inputs,
                                                           model_options=model_options,
                                                           image_pyramid=None,
                                                           weight_decay=l2,
                                                           is_training=tower_setup.is_training,
                                                           fine_tune_batch_norm=not tower_setup.freeze_batchnorm)
    self.outputs = [outputs_to_scales_to_logits["features"]["merged_logits"]]
