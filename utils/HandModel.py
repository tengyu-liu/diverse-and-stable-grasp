import os

import numpy as np
import torch
from manopth.manolayer import ManoLayer


class HandModel:
  def __init__(
    self, 
    n_handcode=6, root_rot_mode='ortho6d', robust_rot=False, flat_hand_mean=False,
    mano_path='data/mano', n_contact=3, scale=120):

    self.scale = scale
    self.n_contact = n_contact
    self.device = torch.device('cuda')
    device = self.device

    if n_handcode == 45:
      self.layer = ManoLayer(root_rot_mode=root_rot_mode, robust_rot=robust_rot, mano_root=mano_path, use_pca=False).to(device)
    else:
      self.layer = ManoLayer(ncomps=n_handcode, root_rot_mode=root_rot_mode, robust_rot=robust_rot, mano_root=mano_path, flat_hand_mean=flat_hand_mean).to(device)
    self.code_length = n_handcode + 6
    if root_rot_mode != 'axisang':
      self.code_length += 3

    self.faces = self.layer.th_faces.detach().cpu().numpy()
    self.num_points = 778
    self.verts_eye = torch.tensor(np.eye(self.num_points)).float().to(device)
    self.n1_mat = self.verts_eye[self.faces[:,0]]   # F x V
    self.n2_mat = self.verts_eye[self.faces[:,1]]   # F x V
    self.n3_mat = self.verts_eye[self.faces[:,2]]   # F x V
    self.fv_total = self.n1_mat.sum(0) + self.n2_mat.sum(0) + self.n3_mat.sum(0) # V

    dense_verts_mult = np.stack(np.meshgrid(
      np.linspace(0, 1, 7), np.linspace(0, 1, 7)), axis=-1).reshape([-1,2])
    self.dense_verts_r1 = torch.from_numpy(np.sqrt(dense_verts_mult[:,0])).float().reshape([1, 1, -1, 1]).to(device)
    self.dense_verts_r2 = torch.from_numpy(dense_verts_mult[:,1]).float().reshape([1, 1, -1, 1]).to(device)
    
    self.z_mean, self.z_std = torch.load(os.path.join(mano_path, 'pose_distrib.pt'))
    self.z_mean = self.z_mean.unsqueeze(0)
    self.z_std = self.z_std.unsqueeze(0)

    self.num_fingertips = 5
    self.fingertip_indices = torch.from_numpy(np.array([763, 350, 462, 573, 690])).cuda().long()

  def prior(self, z):
    if self.code_length == 54:
      return torch.norm((z[:,9:] - self.z_mean) / self.z_std, dim=-1)
    else:
      return torch.norm(z[:,9:], dim=-1)

  def get_vertices(self, hand_code):
    hand_trans = hand_code[:,:3]
    hand_theta = hand_code[:,3:]
    return self.layer(hand_theta)[0] / self.scale + hand_trans.unsqueeze(1)

  def get_vertices_dense(self, hand_code, verts=None):
    if verts is None:
      verts = self.get_vertices(hand_code)    # B x V x 3
    face_verts = verts[:, self.faces]       # B x F x 3 x 3
    # self.dense_vert_r: 49
    dense_verts = (1-self.dense_verts_r1) * face_verts[:,:,[0],:] + \
      self.dense_verts_r1 * (1-self.dense_verts_r2) * face_verts[:,:,[1], :] + \
        self.dense_verts_r1 * self.dense_verts_r2 * face_verts[:,:,[2], :]
    return dense_verts.reshape(verts.shape[0], -1, 3)
   
  def get_surface_normals(self, z=None, verts=None):
    if verts is None:
      if z is None:
        raise ValueError
      else:
        verts = self.get_vertices(z)
    
    B = verts.shape[0]
    V = verts.shape[1]
    
    # get all face verts
    fv1 = verts[:,self.faces[:,0],:]
    fv2 = verts[:,self.faces[:,1],:]
    fv3 = verts[:,self.faces[:,2],:]

    # compute normals
    vn1 = torch.cross((fv1-fv3), (fv2-fv1))   # B x F x 3
    vn2 = torch.cross((fv2-fv1), (fv3-fv2))   # B x F x 3
    vn3 = torch.cross((fv3-fv2), (fv1-fv3))   # B x F x 3

    vn1 = vn1 / torch.norm(vn1, dim=-1, keepdim=True)
    vn2 = vn2 / torch.norm(vn2, dim=-1, keepdim=True)
    vn3 = vn3 / torch.norm(vn3, dim=-1, keepdim=True)

    # aggregate normals
    normals = (torch.einsum('bfn,fv->bvn', vn1, self.n1_mat) + torch.einsum('bfn,fv->bvn', vn2, self.n2_mat) + torch.einsum('bfn,fv->bvn', vn3, self.n3_mat)) / self.fv_total.unsqueeze(0).unsqueeze(-1)
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)
    return normals
  
if __name__ == "__main__":
  import numpy as np
  import random
  import plotly
  import plotly.graph_objects as go

  hand_model = HandModel()
  z = torch.normal(0,1,size=[1,15]).float().to(hand_model.device) * 1e-6
  verts = hand_model.get_vertices(z)
  dense_verts = hand_model.get_vertices_dense(z)

  verts = verts[0].detach().cpu().numpy()
  dense_verts = dense_verts[0].detach().cpu().numpy()
  fti = hand_model.finger15_indices.detach().cpu().numpy()
  print(dense_verts.shape)

  fig = go.Figure(data=[
    go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2], i=hand_model.faces[:,0], j=hand_model.faces[:,1], k=hand_model.faces[:,2]),
    go.Scatter3d(x=verts[fti, 0], y=verts[fti, 1], z=verts[fti, 2], mode='markers', marker=dict(size=5, color='red'))
  ])

  fig.show()
