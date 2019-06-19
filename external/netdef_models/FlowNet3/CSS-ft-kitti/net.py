#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import netdef_slim as nd
from netdef_slim.networks.flownet.flownet_2f_env import FlowNet2f_Environment
from netdef_slim.networks.base_network import BaseNetwork as FlowNet_BaseNetwork
from netdef_slim.architectures.architecture_c import  Architecture_C
from netdef_slim.architectures.architecture_s import  Architecture_S

class Network(FlowNet_BaseNetwork):

    def resample_occ(self, blob, ref):
        resampled = nd.ops.resample(blob, reference=ref)
        return nd.ops.softmax(resampled)

    def make_graph(self, data, include_losses=True):

        pred_config = nd.PredConfig()

        pred_config.add(nd.PredConfigId(type='flow', dir='fwd', offset=0, channels=2, scale=self._scale))
        pred_config.add(nd.PredConfigId(type='occ', dir='fwd', offset=0, channels=2, scale=self._scale))

        nd.log('pred_config:')
        nd.log(pred_config)

        #### Net 1 ####
        with nd.Scope('net1', learn=False, **self.scope_args()):
            arch1 =  Architecture_C(
                      num_outputs=pred_config.total_channels(),
                      disassembling_function=pred_config.disassemble,
                      loss_function = None,
                      conv_upsample=self._conv_upsample
                    )

            out1 = arch1.make_graph(data.img[0], data.img[1])



        #### Net 2 ####
        flow_fwd = out1.final.flow[0].fwd
        upsampled_flow_fwd = nd.ops.differentiable_resample(flow_fwd, reference=data.img[0])
        warped = nd.ops.warp(data.img[1], upsampled_flow_fwd)

        # prepare data for second net
        occ_fwd = self.resample_occ(out1.final.occ[0].fwd, data.img[0])

        input2 = nd.ops.concat(data.img[0], data.img[1],
                               nd.ops.scale(upsampled_flow_fwd, 0.05),
                               warped,
                               occ_fwd
                               )

        pred_config[0].mod_func = lambda x: nd.ops.add(x, nd.ops.resample(flow_fwd, reference=x, type='LINEAR', antialias=False))
        pred_config[1].mod_func = lambda x: nd.ops.add(x, nd.ops.resample(occ_fwd, reference=x, type='LINEAR', antialias=False))

        with nd.Scope('net2', learn=False, **self.scope_args()):

            arch2 = Architecture_S(
                num_outputs=pred_config.total_channels(),
                disassembling_function=pred_config.disassemble,
                loss_function= None,
                conv_upsample=self._conv_upsample
            )
            out2 = arch2.make_graph(input2)



        #### Net 3 ####


        flow_fwd = out2.final.flow[0].fwd
        upsampled_flow_fwd = nd.ops.differentiable_resample(flow_fwd, reference=data.img[0])
        warped = nd.ops.warp(data.img[1], upsampled_flow_fwd)

        # prepare data for third net
        occ_fwd = self.resample_occ(out2.final.occ[0].fwd, data.img[0])

        input3 = nd.ops.concat(data.img[0], data.img[1],
                               nd.ops.scale(upsampled_flow_fwd, 0.05),
                               warped,
                               occ_fwd
                               )

        pred_config.add(nd.PredConfigId(type='mb', dir='fwd', offset=0, channels=2, scale=self._scale))

        pred_config[0].mod_func = lambda x: nd.ops.add(x, nd.ops.resample(flow_fwd, reference=x, type='LINEAR', antialias=False))
        pred_config[1].mod_func = lambda x: nd.ops.add(x, nd.ops.resample(occ_fwd, reference=x, type='LINEAR', antialias=False))

        with nd.Scope('net3', learn=True, **self.scope_args()):

            arch3 = Architecture_S(
                num_outputs=pred_config.total_channels(),
                disassembling_function=pred_config.disassemble,
                loss_function=None,
                conv_upsample=self._conv_upsample,
                exit_after=0,
            )
            out3 = arch3.make_graph(input3, edge_features=data.img[0])

        return out3

net = Network(
	scale=1.0,
        conv_upsample=False
)


def get_env():
    env = FlowNet2f_Environment(net,)
    return env
