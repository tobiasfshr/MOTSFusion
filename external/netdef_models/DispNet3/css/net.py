import netdef_slim as nd
from netdef_slim.networks.dispnet.dispnet_2f_env import DispNet2f_Environment
from netdef_slim.networks.base_network import BaseNetwork as DispNet_BaseNetwork
from netdef_slim.architectures.architecture_c import  Architecture_C
from netdef_slim.architectures.architecture_s import  Architecture_S


class Network(DispNet_BaseNetwork):

    def resample_occ(self, blob, ref):
        resampled = nd.ops.differentiable_resample(blob, reference=ref)
        return nd.ops.softmax(resampled)


    def make_graph(self, data, include_losses=True):
        pred_config = nd.PredConfig()
        pred_config.add(nd.PredConfigId(type='disp',
                                        perspective='L',
                                        channels=1,
                                        scale=self._scale,
                                        mod_func=lambda b: nd.ops.neg_relu(b))
                        )

        pred_config.add(nd.PredConfigId(type='occ',
                                        perspective='L',
                                        channels=2,
                                        scale=self._scale,
                                        )
                        )


        with nd.Scope('net1', learn=False, **self.scope_args()):
            arch1 = Architecture_C(
                num_outputs=pred_config.total_channels(),
                disassembling_function=pred_config.disassemble,
                loss_function= None,
                conv_upsample=self._conv_upsample,
                channel_factor=0.375
            )
            out1 = arch1.make_graph(data.img.L, data.img.R, use_1D_corr=True, single_direction=0)
        disp1 = out1.final.disp.L
        occ1  = self.resample_occ(out1.final.occ.L, data.img.L)

        upsampled_disp1 = nd.ops.differentiable_resample(disp1, reference=data.img.L)
        pred_config[0].mod_func = lambda x: nd.ops.add(x, nd.ops.resample(disp1, reference=x, type='LINEAR', antialias=False))
        pred_config[1].mod_func = lambda x: nd.ops.add(x, nd.ops.resample(occ1, reference=x, type='LINEAR', antialias=False))

        warped = nd.ops.warp(data.img.R, nd.ops.disp_to_flow(upsampled_disp1))

        input2 = nd.ops.concat(data.img.L, data.img.R, nd.ops.scale(upsampled_disp1, 0.05), warped, occ1)

        with nd.Scope('net2', learn=False, **self.scope_args()):
            arch2 = Architecture_S(
                num_outputs=pred_config.total_channels(),
                disassembling_function=pred_config.disassemble,
                loss_function= None,
                conv_upsample=self._conv_upsample,
                channel_factor=0.375
            )
            out2 = arch2.make_graph(input2)
        ## Net 3

        disp2 = out2.final.disp.L
        occ2  = self.resample_occ(out2.final.occ.L, data.img.L)

        upsampled_disp2 = nd.ops.differentiable_resample(disp2, reference=data.img.L)
        pred_config.add(nd.PredConfigId(type='db',
                                        perspective='L',
                                        channels=2,
                                        scale=self._scale))


        pred_config[0].mod_func = lambda x: nd.ops.add(x, nd.ops.resample(disp2, reference=x, type='LINEAR', antialias=False))
        pred_config[1].mod_func = lambda x: nd.ops.add(x, nd.ops.resample(occ2, reference=x, type='LINEAR', antialias=False))

        warped = nd.ops.warp(data.img.R, nd.ops.disp_to_flow(upsampled_disp2))

        input3 = nd.ops.concat(data.img.L, data.img.R, nd.ops.scale(upsampled_disp2, 0.05), warped, occ2)

        with nd.Scope('net3', learn=True, **self.scope_args()):
            arch3 = Architecture_S(
                num_outputs=pred_config.total_channels(),
                disassembling_function=pred_config.disassemble,
                loss_function= None,
                conv_upsample=self._conv_upsample,
                exit_after=0,
                channel_factor=0.375
            )
            out3 = arch3.make_graph(input3, edge_features=data.img.L)
        return out3

net = Network(
    conv_upsample=False,
    scale=1.0,
)



def get_env():
    env = DispNet2f_Environment(net,)
    return env
