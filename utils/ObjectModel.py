import json
import os

import thirdparty.DeepSDF.networks.deep_sdf_decoder as arch
import torch
import trimesh as tm


class DeepSDFModel:
    def __init__(
        self, 
        state_dict_path="data/DeepSDF/2000.pth", 
        predict_normal=False, 
        code_path='data/DeepSDF/Reconstructions/2000/Codes/ShapeNetCore.v2/02876657',
        mesh_path='data/DeepSDF/Reconstructions/2000/Meshes/ShapeNetCore.v2/02876657'):

        self.code_path = code_path
        self.mesh_path = mesh_path

        self.decoder = arch.Decoder(
            latent_size=256, 
            dims=[ 512, 512, 512, 512, 512, 512, 512, 512 ],
            dropout=[0, 1, 2, 3, 4, 5, 6, 7],
            dropout_prob=0.2,
            norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
            latent_in=[4],
            xyz_in_all=False,
            use_tanh=False,
            latent_dropout=False,
            weight_norm=True, 
            predict_normal=predict_normal
        )

        self.device = torch.device('cuda')
        device = self.device

        self.decoder = torch.nn.DataParallel(self.decoder)

        saved_model_state = torch.load(
            state_dict_path
        )
        self.saved_model_epoch = saved_model_state["epoch"]
        self.decoder.load_state_dict(saved_model_state["model_state_dict"])
        self.decoder = self.decoder.module.to(device)
        self.decoder.eval()

        self.codes_ready()

        pass

    def distance(self, obj_code, x):
        # obj_code: B x 256
        # x: B x P x 3
        B = x.shape[0]
        P = x.shape[1]
        obj_code = obj_code.view(B,1,256).repeat(1,P,1).reshape([B*P, 256])
        x = x.reshape([B*P, 3])
        distance = self.decoder(torch.cat([obj_code, x], 1)).reshape([B, P, 1])
        return -distance

    def gradient(self, x, distance, retain_graph=False, create_graph=False, allow_unused=False):
        return torch.autograd.grad([distance.sum()], [x], retain_graph=retain_graph, create_graph=create_graph, allow_unused=allow_unused)[0]

    def closest_point(self, obj_code, x):
        distance = self.distance(obj_code, x)
        gradient = self.gradient(x, distance)
        normal = gradient.clone()
        count = 0
        while torch.abs(distance).mean() > 0.003 and count < 100:
            x = x - gradient * distance * 0.5
            distance = self.distance(obj_code, x)
            gradient = self.gradient(x, distance)
            count += 1
        return x.detach(), normal

    def codes_ready(self):
        _codes = []
        self._meshes = []
        self._mesh_fns = []
        _fns = os.listdir(self.code_path)
        skip = json.JSONDecoder().decode(open('data/DeepSDF/skip.json').read())

        for fn in _fns:
            if fn in skip:
                continue
            _codes.append(torch.load(os.path.join(self.code_path, fn)).squeeze().float().cuda())
            self._meshes.append(tm.load(os.path.join(self.mesh_path, fn[:-4] + '_cd.obj'), force='mesh'))
            self._mesh_fns.append(os.path.join(self.mesh_path, fn[:-3] + 'obj'))

        self.codes = torch.stack(_codes, 0)

    def get_obj_code_random(self, batch_size, code_length=256):
        idx = torch.randint(0, len(self.codes), size=[batch_size], device='cuda')
        return self.codes[idx], idx

    def get_obj_mesh(self, idx):
        return self._meshes[idx]

    def get_obj_mesh_by_code(self, code):
        for i, c in enumerate(self.codes):
            if torch.norm(code - c) < 1e-8:
                return self._meshes[i]

    def get_obj_fn_by_code(self, code):
        for i, c in enumerate(self.codes):
            if torch.norm(code - c) < 1e-8:
                return self._mesh_fns[i]

    def get_grasp_code_random(self, batch_size, code_length):
        code = torch.normal(mean=0, std=1, size=[batch_size, code_length], device='cuda').float()
        return code


class SphereModel:

    def distance(self, radius, x):
        # obj_code: B
        # x: B x P x 3
        # return self.decoder(x, obj_idx)
        return radius - torch.norm(x, dim=-1)

    def gradient(self, x, distance, retain_graph=False, create_graph=False, allow_unused=False):
        return x * -1

    def num_obj(self):
        return 0

