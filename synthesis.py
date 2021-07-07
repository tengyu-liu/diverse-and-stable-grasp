import argparse
import os

import torch
import trimesh as tm
from plotly import graph_objects as go

import utils.visualize_plotly

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--n_contact', default=5, type=int)
parser.add_argument('--max_physics', default=10000, type=int)
parser.add_argument('--max_refine', default=1000, type=int)
parser.add_argument('--hand_model', default='mano', type=str)
parser.add_argument('--obj_model', default='sphere', type=str)
parser.add_argument('--langevin_probability', default=0.85, type=float)
parser.add_argument('--hprior_weight', default=1, type=float)
parser.add_argument('--noise_size', default=0.1, type=float)
parser.add_argument('--mano_path', default='data/mano', type=str)
parser.add_argument('--output_dir', default='synthesis', type=str)
args = parser.parse_args()

from utils.HandModel import HandModel
from utils.Losses import FCLoss
from utils.ObjectModel import DeepSDFModel, SphereModel
from utils.PenetrationModel import PenetrationModel
from utils.PhysicsGuide import PhysicsGuide

# prepare models
if args.obj_model == 'bottle':
    object_model = DeepSDFModel()
    object_code, object_idx = object_model.get_obj_code_random(args.batch_size)
elif args.obj_model == 'sphere':
    object_model = SphereModel()
    object_code = torch.rand(args.batch_size, 1, device='cuda', dtype=torch.float) * 0.2 + 0.1
else:
    raise NotImplementedError()

hand_model = HandModel(
    n_handcode=45,
    root_rot_mode='ortho6d', 
    robust_rot=False,
    flat_hand_mean=False,
    mano_path=args.mano_path, 
    n_contact=args.n_contact)

fc_loss_model = FCLoss(object_model=object_model)
penetration_model = PenetrationModel(hand_model=hand_model, object_model=object_model)

physics_guide = PhysicsGuide(hand_model, object_model, penetration_model, fc_loss_model, args)

accept_history = []

if args.hand_model == 'mano_fingertip':
    num_points = hand_model.num_fingertips
elif args.hand_model == 'mano':
    num_points = hand_model.num_points
else:
    raise NotImplementedError()

z = torch.normal(0, 1, [args.batch_size, hand_model.code_length], device='cuda', dtype=torch.float32, requires_grad=True)
contact_point_indices = torch.randint(0, hand_model.num_points, [args.batch_size, args.n_contact], device='cuda', dtype=torch.long)

# optimize hand pose and contact map using physics guidance
energy, grad, verbose_energy = physics_guide.initialize(object_code, z, contact_point_indices)
linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
accept = ((force_closure < 0.5) * (penetration < 0.02) * (surface_distance < 0.02)).float()
for physics_step in range(args.max_physics):
    energy, grad, z, contact_point_indices, verbose_energy = physics_guide.optimize(energy, grad, object_code, z, contact_point_indices, verbose_energy)
    linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
    accept = ((force_closure < 0.5) * (penetration < 0.02) * (surface_distance < 0.02)).float()
    _accept = accept.sum().detach().cpu().numpy()
    accept_history.append(_accept)
    if physics_step % 100 == 0:
        print('optimize', physics_step, _accept)

for refinement_step in range(args.max_refine):
    energy, grad, z, contact_point_indices, verbose_energy = physics_guide.refine(energy, grad, object_code, z, contact_point_indices, verbose_energy)
    linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
    accept = ((force_closure < 0.5) * (penetration < 0.02) * (surface_distance < 0.02)).float()
    _accept = accept.sum().detach().cpu().numpy()
    accept_history.append(_accept)
    if refinement_step % 100 == 0:
        print('refine', refinement_step, _accept)


os.makedirs('%s/%s-%s-%d-%d'%(args.output_dir, args.hand_model, args.obj_model, args.n_contact, args.batch_size), exist_ok=True)

for a in torch.where(accept)[0]:
    a = a.detach().cpu().numpy()
    hand_verts = physics_guide.hand_model.get_vertices(z)[a].detach().cpu().numpy()
    hand_faces = physics_guide.hand_model.faces
    if args.obj_model == "sphere":
        sphere = tm.primitives.Sphere(radius=object_code[a].detach().cpu().numpy())
        fig = go.Figure([utils.visualize_plotly.plot_hand(hand_verts, hand_faces), utils.visualize_plotly.plot_obj(sphere)])
    else:
        mesh = object_model.get_obj_mesh(object_idx[[a]].detach().cpu().numpy())
        fig = go.Figure([utils.visualize_plotly.plot_hand(hand_verts, hand_faces), utils.visualize_plotly.plot_obj(mesh)])
    fig.write_html('%s/%s-%s-%d-%d/fig-%d.html'%(args.output_dir, args.hand_model, args.obj_model, args.n_contact, args.batch_size, a))

