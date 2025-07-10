import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import time

def positional_encoding(positions, freqs):
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]

def RGBRender(xyz_sampled, viewdirs, features):
    rgb = features
    return rgb

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = inChanel
        self.viewpe = viewpe
        self.n_color = 3

        hidden_dim = 64
        self.layer1 = torch.nn.Linear(self.in_mlpC, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.color_weight = torch.nn.Linear(hidden_dim,self.n_color + 1)
        self.color_basis = torch.nn.Parameter(torch.randn(self.n_color,3))

        torch.nn.init.constant_(self.color_weight.bias, 0)
        
        self.softplus = torch.nn.Softplus()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, pts, features, custom_rgb=None):
        mlp_in = features

        out_1 = torch.relu(self.layer1(mlp_in))
        out_2 = torch.relu(self.layer2(out_1))
        out_3 = self.color_weight(out_2)
        
        c_w = torch.sigmoid(out_3[:,:3])

        rgb = c_w
        
        c_w = c_w / torch.sum(c_w, dim=-1,keepdim=True)
        c_w = (c_w - c_w.min(dim=-1,keepdim=True).values) / (c_w.max(dim=-1,keepdim=True).values - c_w.min(dim=-1,keepdim=True).values)

        sigma = self.softplus(out_3[:,3:]-10).squeeze(-1)

        return rgb, sigma, out_1, out_2, c_w


class BaseModel(torch.nn.Module):
    def __init__(self, aabb, gridSize, device, n_comp = 16,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0],
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus'):
        super(BaseModel, self).__init__()

        self.n_comp = n_comp
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.volumeCache = None
        self.device=device

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio


        self.update_stepSize(gridSize)

        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]


        self.init_svd_volume(gridSize[0], device)
        

        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.n_comp[0], view_pe, featureC).to(device)
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize= torch.LongTensor(gridSize).to(self.device)
        self.units=self.aabbSize / (self.gridSize-1)
        self.stepSize=torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples=int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'n_comp': self.n_comp,

            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,

            'near_far': self.near_far,
            'step_ratio': self.step_ratio,

            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])


    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox


    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]


    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)


    def compute_alpha(self, xyz_locs, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_features(xyz_sampled)
            validsigma = self.renderModule(xyz_sampled, sigma_feature)
            sigma[alpha_mask] = validsigma[1]
        

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha

    def normalize_vector(self, vec):
        norm = torch.norm(vec, p=2, dim=-1, keepdim=True)
        norm = torch.where(norm == 0, torch.tensor(1.0, device=norm.device), norm)
        vec = vec / norm
        return vec

    def enhanced_forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        print_time = False
        if print_time:
            time_start_all = time.time()
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        
        if print_time:
            print(f'Prepare: {(time.time()-time_start_all)*1000}')

        if self.alphaMask is not None:
            if print_time:
                start = time.time()
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            if print_time:
                print(f'Sample alpha: {(time.time()-start)*1000}')
            if print_time:
                print(ray_valid.sum())
            
            if print_time:
                start = time.time()
            ray_valid_new = torch.zeros_like(ray_valid)
            ray_valid_new[ray_valid] = alphas > 0
            ray_valid = ray_valid_new
            
            if print_time:
                print(ray_valid.sum())
            if print_time:
                print(f'Sample alpha: {(time.time()-start)*1000}')

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        color_weights = torch.zeros((*xyz_sampled.shape[:2], self.renderModule.n_color), device=xyz_sampled.device)

        # Default Rendering
        if ray_valid.any():
            xyz_sampled_normalized = self.normalize_coord(xyz_sampled)
            start = time.time()
            if print_time:
                print(ray_valid.sum())
            sigma_feature = self.compute_features(xyz_sampled_normalized[ray_valid])
            if print_time:
                print(f'Sample grid: {(time.time()-start)*1000}')
            start = time.time()
            validsigma = self.renderModule(xyz_sampled_normalized[ray_valid], sigma_feature)
            if print_time:
                print(f'Through MLP: {(time.time()-start)*1000}')
            sigma[ray_valid] = validsigma[1]
            rgb[ray_valid] = validsigma[0]
            color_weights[ray_valid] = validsigma[4]
        
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > 0

        acc_map = torch.sum(weight, -1)


        render_modes = ['surface','none','recolor'] # ENHANCE
        # render_modes = ['default','default','default']
        normal_sample_interval = 1e-4
        rgb_map_surf = torch.ones((xyz_sampled.shape[0], 3), device=xyz_sampled.device)
        rgb_map_surf_mask = torch.zeros((xyz_sampled.shape[0]), dtype=torch.bool, device=xyz_sampled.device)
        dists_final = dists.clone()

        # Enhanced Rendering by Color Palette
        if app_mask.any() and ('depth' in render_modes or 'surface' in render_modes or 'none' in render_modes):            
            for color_index in range(self.renderModule.n_color):
                color_mask = torch.logical_and(app_mask, torch.argmax(color_weights, dim=-1) == color_index)
                render_mode = render_modes[color_index]

                if render_mode == 'default' and color_mask.any():
                    pass

                if render_mode == 'recolor' and color_mask.any():
                    rgb[color_mask] = color_weights[color_mask][:,color_index].unsqueeze(-1) * torch.tensor([0.7,0.4,0.3]).cuda()
                    
                # Depth Enhancement
                if render_mode == 'depth' and color_mask.any():
                    p0 = xyz_sampled[color_mask]
                    ray_dir = viewdirs[color_mask]
                    rt = dists[color_mask].unsqueeze(-1) * 2
                    p1 = p0 + rt * ray_dir

                    sigma_feature = self.compute_features(p1)
                    validsigma = self.renderModule(p1, sigma_feature)

                    # max_dsteps = 5
                    max_dsteps = 4
                    adapt_sampling_thres = 0.01
                    diff = torch.abs(sigma[color_mask] - validsigma[1])
                    d_steps = diff / adapt_sampling_thres
                    d_steps = torch.clamp(d_steps, 0, max_dsteps)
                    d_steps_save = d_steps

                    w = torch.zeros((*(p0).shape[:1], 1), device=p0.device)
                    c = torch.zeros((*(p0).shape[:1], 3), device=p0.device)

                    one = torch.ones_like(d_steps_save)

                    for ahc in range(max_dsteps):
                        adapt_mask = torch.logical_and((d_steps > ahc), (d_steps > 1))
                        if not adapt_mask.any():
                            break
                        rt = one[adapt_mask] * ahc / d_steps[adapt_mask]
                        rt = rt.unsqueeze(-1)
                        pi = (1.0 - rt) * p0[adapt_mask] + rt * p1[adapt_mask]
                        pi = self.normalize_coord(pi)
                        features = self.compute_features(pi)
                        outputs = self.renderModule(pi, features)
                        output_sigma = outputs[1]
                        output_rgb = outputs[0]
                        output_cw = outputs[4]
                        output_sigma[torch.argmax(output_cw, dim=-1) != color_index] = 0

                        omda = (1.0 - w[adapt_mask])
                        c[adapt_mask] += omda * output_rgb.clamp(0,1) * output_sigma.clamp(0,1).unsqueeze(-1)
                        w[adapt_mask] += omda * output_sigma.clamp(0,1).unsqueeze(-1)

                    dists_final[color_mask] = dists[color_mask] / 2 * d_steps

                    sigma[color_mask] = w.squeeze(-1)
                    rgb[color_mask] = c

                    for plane_idx in range(3):
                        self.plane[plane_idx] = self.smooth_planes[plane_idx]

                    normals = torch.zeros_like(rgb[color_mask])
                    for dim_i in range(normals.size(-1)):
                        pos = self.normalize_coord(torch.clone(p0))
                        pos[:,dim_i] = pos[:,dim_i] + normal_sample_interval
                        pos_sigma = self.renderModule(pos, self.compute_features(pos))[1]
                        pos_alpha =  1 - torch.exp(-pos_sigma)

                        neg = self.normalize_coord(torch.clone(p0))
                        neg[:,dim_i] = neg[:,dim_i] - normal_sample_interval
                        neg_sigma = self.renderModule(neg, self.compute_features(neg))[1]
                        neg_alpha =  1 - torch.exp(-neg_sigma)

                        normals[:,dim_i] = pos_alpha - neg_alpha

                    for plane_idx in range(3):
                        self.plane[plane_idx] = self.real_planes[plane_idx]

                    N = self.normalize_vector(normals)
                    V = rays_chunk[:,3:6].unsqueeze(1).repeat(1,color_mask.size(1),1)[color_mask]
                    V = -self.normalize_vector(V)
                    L = rays_chunk[:, :3].unsqueeze(1).repeat(1,color_mask.size(1),1)[color_mask]
                    L = self.normalize_vector(L)
                    H = L + V
                    H = self.normalize_vector(H)

                    dot_neg_LN = torch.sum(-L * N, dim=-1)
                    rev = dot_neg_LN > 0
                    N[rev] = -N[rev]

                    dot_LN = torch.sum(L * N, dim=-1).unsqueeze(-1)
                    dot_NH = torch.sum(H * N, dim=-1).unsqueeze(-1)

                    shininess = 50
                    ambient_factor = 0.2
                    specular_factor = 1

                    rgb[color_mask] = rgb[color_mask] * (dot_LN + ambient_factor) + torch.pow(dot_NH, shininess) * specular_factor
                    rgb[color_mask] = torch.clamp(rgb[color_mask],0,1)

                
                
                if render_mode == 'surface' and color_mask.any():
                    thres = 0.01
                    temp_sigma = sigma.clone()
                    temp_alpha, _, _ = raw2alpha(temp_sigma, dists * self.distance_scale)
                   
                    curr_alpha = temp_alpha[:,:-1]
                    next_alpha = temp_alpha[:,1:]
                    curr_cond = curr_alpha < thres
                    next_cond = next_alpha > thres
                    cond_a = torch.logical_and(curr_cond, next_cond)
                    curr_cond = curr_alpha > thres
                    next_cond = next_alpha < thres
                    cond_b = torch.logical_and(curr_cond, next_cond)
                    cond = torch.logical_or(cond_a, cond_b)

                    mask = cond
                    mask = torch.logical_and(mask, color_mask[:,:-1])
                    cond = cond[mask]

                    rgb_map_surf_mask[torch.any(mask,-1)] = True
                    
                    if rgb_map_surf_mask.any():
                        xyz_sampled_normalized = self.normalize_coord(xyz_sampled)
                        curr_xyz = xyz_sampled_normalized[:,:-1]
                        next_xyz = xyz_sampled_normalized[:,1:]
                        centroids = (next_xyz - curr_xyz) * 0.5 + curr_xyz
                        start_xyz = rays_chunk[:, :3].unsqueeze(1).expand(-1,curr_xyz.size(1),-1)
                        start_xyz = self.normalize_coord(start_xyz)
                        d_to_start = curr_xyz - start_xyz
                        
                        d_map = torch.zeros_like(mask, dtype=torch.float)
                        d_map[mask] = torch.sqrt(torch.sum(d_to_start[mask] ** 2, -1))
                        
                        d_map[d_map == 0] = float('inf')
                        d_map, min_indices = torch.min(d_map, dim=-1)
                        
                        all_indices = torch.arange(xyz_sampled.size(1), device=xyz_sampled.device)
                        all_indices = all_indices.repeat(xyz_sampled.size(0), 1)
                        all_min_indices = min_indices.unsqueeze(1).repeat(1, xyz_sampled.size(1))
                        
                        back_mask = all_indices > all_min_indices
                        cover_mask = torch.any(mask, dim=-1, keepdim=True)
                        back_mask = torch.logical_and(back_mask, cover_mask)
                        
                        
                        sigma[back_mask] = 0

                        min_indices = min_indices.unsqueeze(1).unsqueeze(2)
                        closest_dists = torch.gather(dists[:,:-1].unsqueeze(-1), 1, min_indices).squeeze(1)
                        min_indices = min_indices.expand(-1,-1,3)
                        closest_centroids = torch.gather(centroids, 1, min_indices).squeeze(1)
                        closest_xyz = torch.gather(curr_xyz, 1, min_indices).squeeze(1)
                    
                        # Before Normal Change Planes
                        for plane_idx in range(3):
                            self.plane[plane_idx] = self.smooth_planes[plane_idx]

                        all_normals = torch.zeros_like(closest_centroids)
                        if True:
                            normals = torch.zeros_like(closest_centroids)
                            for dim_i in range(normals.size(-1)):
                                pos = torch.clone(closest_centroids)
                                pos[:,dim_i] = pos[:,dim_i] + normal_sample_interval
                                results = self.renderModule(pos, self.compute_features(pos))
                                pos_sigma = results[1]
                                pos_cw = results[4]
                                pos_sigma *= pos_cw[:, color_index]
                                pos_alpha = pos_sigma
                                
                                neg = torch.clone(closest_centroids)
                                neg[:,dim_i] = neg[:,dim_i] - normal_sample_interval
                                results = self.renderModule(neg, self.compute_features(neg))
                                neg_sigma = results[1]
                                neg_cw = results[4]
                                neg_sigma *= neg_cw[:, color_index]
                                neg_alpha = neg_sigma
                                
                                normals[:,dim_i] = pos_alpha - neg_alpha
                            normals = self.normalize_vector(normals)
                            all_normals += normals
                        normals = all_normals
                        
                        ### Lighting ###
                        N = normals
                        V = rays_chunk[:, 3:6].unsqueeze(1).expand(-1,curr_xyz.size(1),-1)[:,0,:]
                        V = -self.normalize_vector(V)
                        L = rays_chunk[:, :3].unsqueeze(1).expand(-1,curr_xyz.size(1),-1)[:,0,:]
                        L = self.normalize_vector(L)
                        H = L + V
                        H = self.normalize_vector(H)

                        
                        dot_neg_LN = torch.sum(-L * N, dim=-1)
                        rev = dot_neg_LN > 0
                        N[rev] = -N[rev]

                        ### Ambient Occlusion ###
                        kernel_normal = torch.tensor([[0.,0.,1.]]).to(self.device)
                        v = self.normalize_vector(torch.cross(N, kernel_normal, dim=-1))
                        c = torch.sum(N * kernel_normal, dim=-1)

                        k_mat = torch.zeros((N.size(0), 3, 3)).to(self.device)
                        k_mat[:, 0, 1] = -v[:, 2]
                        k_mat[:, 0, 2] = v[:, 1]
                        k_mat[:, 1, 0] = v[:, 2]
                        k_mat[:, 1, 2] = -v[:, 0]
                        k_mat[:, 2, 0] = -v[:, 1]
                        k_mat[:, 2, 1] = v[:, 0]
                        I = torch.eye(3).to(self.device).unsqueeze(0).repeat(N.size(0), 1, 1)
                        v_norm_sq = torch.sum(v * v, dim=1)
                        bmm_k_mat = torch.bmm(k_mat, k_mat)
                        mul = ((1 - c) / (v_norm_sq + 1e-10))
                        bmm_k_mat = bmm_k_mat * mul.view(-1,1,1)
                        rotation_matrices = I + k_mat + bmm_k_mat
                        
                        kernel = self.kernel.unsqueeze(0).expand(N.size(0),-1,-1)
                        kernel_x = kernel[:, :, 0]
                        kernel_y = kernel[:, :, 1]
                        kernel_z = kernel[:, :, 2]
                        rotation_x = rotation_matrices[:, :, 0]
                        rotation_y = rotation_matrices[:, :, 1]
                        rotation_z = rotation_matrices[:, :, 2]
                        
                        new_kernel_x = kernel_x * rotation_x[:,0].view(-1,1) + kernel_y * rotation_x[:,1].view(-1,1) + kernel_z * rotation_x[:,2].view(-1,1)
                        new_kernel_y = kernel_x * rotation_y[:,0].view(-1,1) + kernel_y * rotation_y[:,1].view(-1,1) + kernel_z * rotation_y[:,2].view(-1,1)
                        new_kernel_z = kernel_x * rotation_z[:,0].view(-1,1) + kernel_y * rotation_z[:,1].view(-1,1) + kernel_z * rotation_z[:,2].view(-1,1)
                        kernel = torch.stack([new_kernel_x,new_kernel_y,new_kernel_z], dim=-1)
                        
                        ao_sample_xyz = closest_xyz.clone()
                        ao_sample_xyz = ao_sample_xyz.unsqueeze(1).expand(-1,self.ao_n_sample,-1)
                        ao_sample_xyz = ao_sample_xyz + kernel.unsqueeze(0)
                        ao_sample_xyz = ao_sample_xyz.view(-1,3)
                        ao_render_outputs = self.renderModule(ao_sample_xyz, self.compute_features(ao_sample_xyz))
                        ao_color_weights = ao_render_outputs[4]
                        ao_color_mask = torch.argmax(ao_color_weights, dim=-1) != color_index
                        ao_sample_sigma = ao_render_outputs[1]
                        ao_sample_sigma *= ao_color_weights[:, color_index]
                        closest_dists = closest_dists.expand(-1,self.ao_n_sample)
                        
                        ao_sample_sigma = ao_sample_sigma.view(-1,self.ao_n_sample)
                        ao_sample_alpha = 1. - torch.exp(-ao_sample_sigma*closest_dists*self.distance_scale)
                        ao_mask = ao_sample_alpha > thres
                        
                        ao_mask = torch.sum(ao_mask, dim=-1)
                        ao_mask = ao_mask / self.ao_n_sample
                        ao_mask = ao_mask.unsqueeze(-1)
                        ao_mask = ao_mask * 1.5
                        ao_mask = ao_mask.clamp(0.0, 1.0)

                        # After Normal Change Planes Back
                        for plane_idx in range(3):
                            self.plane[plane_idx] = self.real_planes[plane_idx]

                        dot_LN = torch.sum(L * N, dim=-1).unsqueeze(-1)
                        dot_NH = torch.sum(H * N, dim=-1).unsqueeze(-1)
                        shininess = 50
                        spec_amnt = torch.pow(dot_NH, shininess)

                        diffuse_color = torch.FloatTensor([0.95,0.95,0.85]).to(self.device).unsqueeze(0).expand(N.size(0),-1)
                        out = (diffuse_color * dot_LN * 1.0) + (spec_amnt - ao_mask).view(-1,1).repeat(1, 3)
                        
                        rgb_map_surf[rgb_map_surf_mask] = out[rgb_map_surf_mask]
                        
                if (render_mode == 'surface' or render_mode == 'none') and color_mask.any():
                    sigma[color_mask] = 0
                    
            alpha, weight, _ = raw2alpha(sigma.squeeze(-1), dists_final * self.distance_scale)
            acc_map = torch.sum(weight, -1)
        
        
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map[~rgb_map_surf_mask] = rgb_map[~rgb_map_surf_mask] + (1. - acc_map[..., None][~rgb_map_surf_mask])
            rgb_map[rgb_map_surf_mask] = rgb_map[rgb_map_surf_mask] + (1 - acc_map[..., None][rgb_map_surf_mask]) * rgb_map_surf[rgb_map_surf_mask]

        
        rgb_map = rgb_map.clamp(0,1)
        

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        
        return rgb_map, depth_map
        
    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            ray_valid_new = torch.zeros_like(ray_valid)
            ray_valid_new[ray_valid] = alphas > 0
            ray_valid = ray_valid_new
            

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        volval = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        color_weights = torch.zeros((*xyz_sampled.shape[:2], self.renderModule.n_color), device=xyz_sampled.device)

        if True and ray_valid.any():
            xyz_sampled_normalized = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_features(xyz_sampled_normalized[ray_valid])
            validsigma = self.renderModule(xyz_sampled_normalized[ray_valid], sigma_feature)
            
            sigma[ray_valid] = validsigma[1]
            rgb[ray_valid] = validsigma[0]
            
            color_weights[ray_valid] = validsigma[4]


        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres
        
        acc_map = torch.sum(weight, -1)

        
        

        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        
        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight

class ReVolVE(BaseModel):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(ReVolVE, self).__init__(aabb, gridSize, device, **kargs)


    def init_svd_volume(self, res, device):
        self.plane, self.line = self.init_one_svd(self.n_comp, self.gridSize, 0.1, device)
        print(self.n_comp)
        self.basis_mat = torch.nn.Linear(sum(self.n_comp), self.n_comp[0], bias=False).to(device)

    def create_volume_cache(self):
        self.volumeCache = torch.nn.Parameter(torch.zeros((1+self.renderModule.n_color, self.gridSize[2], self.gridSize[1], self.gridSize[0]))).to(self.device)
        tensor_size_in_bytes = self.volumeCache.element_size() * self.volumeCache.numel()
        tensor_size_in_MB = tensor_size_in_bytes / (1024 ** 2)
        print(f"Volume Cache created, size in VRAM: {tensor_size_in_MB:.2f} MB")

    def sample_volume_cache(self, xyz_sampled):
        coordinate = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        F.grid_sample(self.volumeCache, coordinate, align_corners=True).view(-1, *xyz_sampled.shape[:1])

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    
    

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.line, 'lr': lr_init_spatialxyz}, {'params': self.plane, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.line)
    
    def L1(self):
        total = 0
        for idx in range(len(self.plane)):
            total = total + torch.mean(torch.abs(self.plane[idx])) + torch.mean(torch.abs(self.line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss(self, reg):
        total = 0
        for idx in range(len(self.plane)):
            total = total + reg(self.plane[idx]) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3
        return total


    def compute_features(self, xyz_sampled):

        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.plane)):
            plane_coef_point.append(F.grid_sample(self.plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True, mode='bilinear').view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True, mode='bilinear').view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))


        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.plane, self.line = self.up_sampling_VM(self.plane, self.line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.line[i] = torch.nn.Parameter(
                self.line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.plane[i] = torch.nn.Parameter(
                self.plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

