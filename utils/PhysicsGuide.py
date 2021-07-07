import torch

from .EMA import EMA


class PhysicsGuide:
    def __init__(self, hand_model, object_model, penetration_model, fc_loss_model, args):
        self.epsilon = 1e-4
        self.hand_model = hand_model
        self.object_model = object_model
        self.penetration_model = penetration_model
        self.fc_loss_model = fc_loss_model
        self.args = args
        self.grad_ema = EMA(0.98)
        self.ones = torch.ones([self.args.batch_size, self.hand_model.num_points], device='cuda') # B x V
        self.arange = torch.arange(self.args.batch_size).cuda()
        self.rejection_count = torch.zeros([self.args.batch_size, 1], device='cuda', dtype=torch.long)

    def initialize(self, object_code, z, contact_point_indices):
        linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = self.compute_energy(object_code, z, contact_point_indices, verbose=True)
        energy = linear_independence + force_closure + surface_distance + penetration + z_norm + normal_alignment
        grad = torch.autograd.grad(energy.sum(), z)[0]
        self.grad_ema.apply(grad)
        return energy, grad, [linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment]

    def compute_energy(self, object_code, z, contact_point_indices, verbose=False):
        hand_verts = self.hand_model.get_vertices(z)
        contact_point = torch.gather(hand_verts, 1, torch.tile(contact_point_indices.unsqueeze(-1), [1,1,3]))
        contact_distance = self.object_model.distance(object_code, contact_point)
        contact_normal = self.object_model.gradient(contact_point, contact_distance, create_graph=True, retain_graph=True)
        contact_normal += torch.normal(0, self.epsilon, contact_normal.shape, device=contact_normal.device, dtype=contact_normal.dtype)
        contact_normal = contact_normal / torch.norm(contact_normal, dim=-1, keepdim=True)
        hand_normal = self.hand_model.get_surface_normals(verts=hand_verts)
        hand_normal = torch.gather(hand_normal, 1, torch.tile(contact_point_indices.unsqueeze(-1), [1,1,3]))
        hand_normal += torch.normal(0, self.epsilon, hand_normal.shape, device=hand_normal.device, dtype=hand_normal.dtype)
        hand_normal = hand_normal / torch.norm(hand_normal, dim=-1, keepdim=True)    
        normal_alignment = 1 - ((hand_normal * contact_normal).sum(-1) + 1).sum(-1) / self.args.n_contact
        linear_independence, force_closure = self.fc_loss_model.fc_loss(contact_point, contact_normal, object_code)
        surface_distance = self.fc_loss_model.dist_loss(object_code, contact_point)
        penetration = self.penetration_model.get_penetration(object_code, z) * 10  # B x V
        hand_prior = self.hand_model.prior(z) * self.args.hprior_weight
        if verbose:
            return linear_independence, force_closure, surface_distance.sum(1), penetration.sum(1), hand_prior, normal_alignment
        else:
            return linear_independence + force_closure + surface_distance.sum(1) + penetration.sum(1) + hand_prior + normal_alignment

    def get_stepsize(self, energy):
        return 0.02600707 + energy.unsqueeze(1) * 0.03950357 * 1e-3

    def get_temperature(self, energy):
        return 0.02600707 + energy * 0.03950357

    def optimize(self, energy, grad, object_code, z, contact_point_indices, verbose_energy):
        linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
        batch_size = len(energy)
        step_size = self.get_stepsize(energy)
        temperature = self.get_temperature(energy)
        switch = torch.rand([batch_size, 1], device='cuda')
        # update z by langevin
        noise = torch.normal(mean=0, std=self.args.noise_size, size=z.shape, device='cuda', dtype=torch.float) * step_size
        new_z = z - 0.5 * grad / self.grad_ema.average.unsqueeze(0) * step_size * step_size + noise
        # linear_independence, force_closure, surface_distance, penetration, hand_prior, normal_alignment = self.compute_energy(object_code, new_z, contact_point_indices, verbose=True)
        # print('linear_independence', linear_independence.mean().detach().cpu().numpy())
        # print('force_closure', force_closure.mean().detach().cpu().numpy())
        # print('surface_distance', surface_distance.mean().detach().cpu().numpy())
        # print('penetration', penetration.mean().detach().cpu().numpy())
        # print('hand_prior', hand_prior.mean().detach().cpu().numpy())
        # print('normal_alignment', normal_alignment.mean().detach().cpu().numpy())
        # exit()
        # update contact point by random sampling
        new_contact_point_indices = contact_point_indices.clone()
        update_indices = torch.randint(0, self.args.n_contact, size=[self.args.batch_size], device='cuda')
        prob = self.ones.clone()
        prob[torch.unsqueeze(self.arange, 1), contact_point_indices] = 0
        # sample update_to indices
        if self.args.hand_model == 'mano_fingertip':
            update_to = torch.randint(0, self.hand_model.num_fingertips, size=[self.args.batch_size], device='cuda')
            update_to = self.hand_model.fingertip_indices[update_to]
        else:
            update_to = torch.randint(0, self.hand_model.num_points, size=[self.args.batch_size], device='cuda')
        new_contact_point_indices[self.arange, update_indices] = update_to
        # merge by switch
        update_H = ((switch < self.args.langevin_probability) * (self.rejection_count < 2))
        new_z = new_z * update_H + z * ~update_H
        new_contact_point_indices = new_contact_point_indices * (~update_H) + contact_point_indices * update_H
        # compute new energy
        new_linear_independence, new_force_closure, new_surface_distance, new_penetration, new_z_norm, new_normal_alignment = self.compute_energy(object_code, new_z, new_contact_point_indices, verbose=True)
        new_energy = new_linear_independence + new_force_closure + new_surface_distance + new_penetration + new_z_norm + new_normal_alignment
        new_grad = torch.autograd.grad(new_energy.sum(), new_z)[0]
        # accept by Metropolis-Hasting algorithm
        with torch.no_grad():
            # metropolis-hasting
            alpha = torch.rand(self.args.batch_size, device='cuda', dtype=torch.float)
            accept = alpha < torch.exp((energy - new_energy) / temperature)
            z[accept] = new_z[accept]
            contact_point_indices[accept] = new_contact_point_indices[accept]
            energy[accept] = new_energy[accept]
            grad[accept] = new_grad[accept]
            linear_independence[accept] = new_linear_independence[accept]
            force_closure[accept] = new_force_closure[accept]
            surface_distance[accept] = new_surface_distance[accept]
            penetration[accept] = new_penetration[accept]
            z_norm[accept] = new_z_norm[accept]
            normal_alignment[accept] = new_normal_alignment[accept]
            self.rejection_count[accept] = 0
            self.rejection_count[~accept] += 1
            self.grad_ema.apply(grad)
        # print('delta-z: %f delta-i: %f accept: %f'%(torch.norm(z - old_z), torch.norm(contact_point_indices.float() - old_ind.float()), accept.sum()))
        return energy, grad, z, contact_point_indices, [linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment]

    def refine(self, energy, grad, object_code, z, contact_point_indices, verbose_energy):
        linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment = verbose_energy
        step_size = 0.1
        temperature = 1e-3
        # update z by langevin
        noise = torch.normal(mean=0, std=self.args.noise_size, size=z.shape, device='cuda', dtype=torch.float) * step_size
        new_z = z - 0.5 * grad / self.grad_ema.average.unsqueeze(0) * step_size * step_size + noise
        new_linear_independence, new_force_closure, new_surface_distance, new_penetration, new_z_norm, new_normal_alignment = self.compute_energy(object_code, new_z, contact_point_indices, verbose=True)
        new_energy = new_linear_independence + new_force_closure + new_surface_distance + new_penetration + new_z_norm + new_normal_alignment
        new_grad = torch.autograd.grad(new_energy.sum(), new_z)[0]
        self.grad_ema.apply(grad)
        with torch.no_grad():
            # metropolis-hasting
            alpha = torch.rand(self.args.batch_size, device='cuda', dtype=torch.float)
            accept = alpha < torch.exp((energy - new_energy) / temperature)
            z[accept] = new_z[accept]
            energy[accept] = new_energy[accept]
            grad[accept] = new_grad[accept]
            linear_independence[accept] = new_linear_independence[accept]
            force_closure[accept] = new_force_closure[accept]
            surface_distance[accept] = new_surface_distance[accept]
            penetration[accept] = new_penetration[accept]
            z_norm[accept] = new_z_norm[accept]
            normal_alignment[accept] = new_normal_alignment[accept]
            self.grad_ema.apply(grad)
        return energy, grad, z, contact_point_indices, [linear_independence, force_closure, surface_distance, penetration, z_norm, normal_alignment]
