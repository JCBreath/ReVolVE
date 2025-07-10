
import os
from tqdm.auto import tqdm
from opt import config_parser

import json, random
from renderer import *
from utils import *
import datetime

from data_loader import BlenderDataset
import sys

from math import pi, sin, cos, radians
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.transforms.functional import adjust_sharpness


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast
enhanced_renderer = EnhancementRenderer

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]

@torch.no_grad()
def export(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    revolve = eval(args.model_name)(**kwargs)
    revolve.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    export_dir = f'{logfolder}/exported_files'
    os.makedirs(export_dir, exist_ok=True)

    res = 512
    gridSize = (res,res,res)
    samples = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, gridSize[0]),
        torch.linspace(0, 1, gridSize[1]),
        torch.linspace(0, 1, gridSize[2]),
    ), -1).to(revolve.device)

    aabb = torch.FloatTensor([(-1,-1,-1), (1,1,1)])
    dense_xyz = aabb[0].cuda() * (1-samples) + aabb[1].cuda() * samples
  
    volume = torch.zeros(gridSize)

    for color_index in range(3):
        color_volume = torch.zeros(gridSize)
        for i in range(gridSize[0]):
            batch_xyz = dense_xyz[i].view(-1,3)
            alpha = revolve.alphaMask.sample_alpha(batch_xyz)
            alpha_mask = alpha > 0
            if not alpha_mask.any():
                continue
            batch_sigma = torch.zeros_like(batch_xyz[...,0])
            batch_cw = torch.zeros_like(batch_xyz)
            batch_rgb = torch.zeros_like(batch_xyz)
            normalized_xyz = revolve.normalize_coord(batch_xyz[alpha_mask])
            results = revolve.renderModule(normalized_xyz, revolve.compute_features(normalized_xyz))
            batch_rgb[alpha_mask] = results[0]
            batch_sigma[alpha_mask] = results[1]
            batch_cw[alpha_mask] = results[4]
            batch_sigma[torch.argmax(batch_rgb, dim=-1) != color_index] = 0
            batch_alpha = (1. - torch.exp(-batch_sigma * (1/res)))
            color_volume[i] = batch_alpha.view(gridSize[1:]).cpu()

        # print(color_volume.max(), color_volume.min())
        if color_volume.max() > 0:
            try:
                if args.mode == 'export_mesh':
                    convert_sdf_samples_to_ply(color_volume, f'{export_dir}/mesh_{color_index}.ply',bbox=aabb, level=0.0001)
                if args.mode == 'export_volume':
                    color_volume.numpy().tofile(f'{export_dir}/volume_{color_index}.raw', format='<f')
            except Exception as e:
                pass
    
    print(f'Exported files to {export_dir}.')


@torch.no_grad()
def render_test(args):
    dataset = BlenderDataset
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    revolve = eval(args.model_name)(**kwargs)
    revolve.load(ckpt)
    
    logfolder = os.path.dirname(args.ckpt)

    if args.enhanced:
        # Apply upsampling
        n_voxels = 134217728
        reso_cur = N_to_reso(n_voxels, revolve.aabb)
        nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
        revolve.upsample_volume_grid(reso_cur)

        revolve.real_planes = []
        revolve.smooth_planes = []
        for plane_idx in range(3):
            revolve.real_planes.append(revolve.plane[plane_idx].clone())
            revolve.smooth_planes.append(bilateral_filter(revolve.plane[plane_idx].clone(), 15, 5, 1))
            

    # Apply AO
    revolve.ao_n_sample = 256
    revolve.kernel = torch.rand(revolve.ao_n_sample, 3).to(revolve.device)

    revolve.kernel = revolve.kernel ** 1.5
    revolve.kernel = revolve.kernel * 2 - 1

    revolve.kernel = revolve.kernel * 0.1
    revolve.kernel[:, -1] = torch.abs(revolve.kernel[:, -1])
    
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    
    if not args.enhanced:
        render_dir = f'{logfolder}/render_output'
        os.makedirs(render_dir, exist_ok=True)
        evaluation(test_dataset,revolve, args, renderer, render_dir,
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
    else:
        render_dir = f'{logfolder}/enhanced_output'
        enhanced_evaluation(test_dataset,revolve, args, enhanced_renderer, render_dir,
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
                                
def reconstruction(args):
    # init dataset
    dataset = BlenderDataset
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb = args.n_lamb
    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    
    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)

    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))


    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        revolve = eval(args.model_name)(**kwargs)
        revolve.load(ckpt)
    else:
        revolve = eval(args.model_name)(aabb, reso_cur, device,
                    n_comp=n_lamb, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)


    grad_vars = revolve.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))


    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = revolve.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight = args.TV_weight
    tvreg = TVLoss()
    print(f"initial TV_weight: {TV_weight}")


    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, revolve, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = revolve.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
        if L1_reg_weight > 0:
            loss_reg_L1 = revolve.L1()
            total_loss += L1_reg_weight*loss_reg_L1

        if TV_weight>0:
            TV_weight *= lr_factor
            loss_tv = revolve.TV_loss(tvreg) * TV_weight
            total_loss = total_loss + loss_tv

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(test_dataset,revolve, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
           
        if iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = revolve.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                revolve.shrink(new_aabb)
                
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays,allrgbs = revolve.filtering_rays(allrays,allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, revolve.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            revolve.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = revolve.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        
    revolve.save(f'{logfolder}/{args.expname}.th')

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(f"Running in mode: {args.mode}")
    if args.mode == 'train':
        reconstruction(args)
    elif args.mode == 'render_test':
        print(f"Enhanced: {args.enhanced}")
        render_test(args)
    elif args.mode == 'export_mesh' or args.mode == 'export_volume':
        export(args)

