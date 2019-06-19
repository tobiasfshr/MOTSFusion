from file_io.io_utils import writeFloat, writeFlow
import os
from importlib import reload


def compute_flow_and_disp(save_path, data_path, model_path_disp, model_path_flow):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_list = sorted(os.listdir(data_path + 'image_2'))

    for i in range(len(file_list)):
        import netdef_slim as nd
        img_t0_L = data_path + 'image_2/' + file_list[i]
        img_t0_R = data_path + 'image_3/' + file_list[i]

        dispnet_controller = nd.load_module(model_path_disp + 'controller.py').Controller()
        data = dispnet_controller.net_actions(net_dir=model_path_disp).eval(img_t0_L, img_t0_R)
        disp = data['disp.L'][0, 0, :, :]
        writeFloat(save_path + file_list[i] + '.disp.L.float3', disp)

        flownet_controller = nd.load_module(model_path_flow + 'controller.py').Controller()
        if i < len(file_list) - 1:
            img_t1_L = data_path + 'image_2/' + file_list[i + 1]
            flow_fwd = flownet_controller.net_actions(net_dir=model_path_flow).eval(img_t0_L, img_t1_L)
            for key, value in flow_fwd.items():
                if 'flow' in key:
                    writeFlow(save_path + file_list[i] + '.' + key + '.flo',  value[0,:,:,:].transpose(1,2,0))
                else:
                    writeFloat(save_path + file_list[i] + '.' + key + '.float3',  value[0,:,:,:].transpose(1,2,0))

            flow_bwd = flownet_controller.net_actions(net_dir=model_path_flow).eval(img_t1_L, img_t0_L)
            for key, value in flow_bwd.items():
                if 'flow' in key:
                    writeFlow(save_path + file_list[i] + '.' + key.replace('fwd', 'bwd') + '.flo',  value[0,:,:,:].transpose(1,2,0))
                else:
                    writeFloat(save_path + file_list[i] + '.' + key.replace('fwd', 'bwd') + '.float3',  value[0,:,:,:].transpose(1,2,0))
