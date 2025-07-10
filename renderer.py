import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from model import ReVolVE, raw2alpha, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
import time
import math
from kornia import create_meshgrid
# from torchvision.transforms import GaussianBlur

# from kornia.filters import bilateral_blur


class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta) # x
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta) # y
        camera_origins[:, 1:2] = radius*torch.cos(phi) # z

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = normalize_vecs(lookat_position - camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)
    
    @staticmethod
    def sample_point(camera_origins, lookat_position):

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = normalize_vecs(lookat_position - camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)

def OctreeRender_trilinear_fast(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):

    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
    
        rgb_map, depth_map = tensorf(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
    
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None


def EnhancementRenderer(rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False, device='cuda'):
    rgbs = torch.zeros(rays.shape[0], 3).to(device)
    
    N_rays_all = rays.shape[0]
    if N_rays_all < chunk:
        chunk = N_rays_all
    
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)

        rgb_map, depth_map = tensorf.enhanced_forward(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray, N_samples=N_samples)
        rgbs[chunk_idx * chunk:(chunk_idx + 1) * chunk] = rgb_map
        
    return rgbs, None, None, None, None
    
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    # print(K,R,t)

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def get_ray_directions(H, W, focal, center=None):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)  # (H, W, 3)

    return directions


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))


def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))

    return cam2world

def generate_rays(img_wh=(512,512), xyz=(0,0,0)):
    # Make Transform
    w, h = img_wh
    x, y, z = xyz

    lookat_point = (0, 0, 0)
    camera_pivot = torch.tensor(lookat_point)
    focal_length = (1/2) / math.tan(15/180*math.pi)
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])

    directions = get_ray_directions(1, 1, [focal_length,focal_length])
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    camera_origins = torch.FloatTensor([x,y,z]).view(1,3)  * 4.0311
    forward_cam2world_pose = LookAtPoseSampler.sample_point(camera_origins, camera_pivot)
    c2w = forward_cam2world_pose

    swap_row = torch.FloatTensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    mask = torch.FloatTensor([[-1,1,1,-1],[1,-1,-1,1],[1,-1,-1,1],[1,1,1,1]])
    blender2opencv = torch.FloatTensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    up = torch.FloatTensor([0, 1, 0])

    c2w = create_cam2world_matrix(normalize_vecs(camera_pivot + 1e-8 - camera_origins), camera_origins)
    c2w = c2w[0]
    c2w = swap_row @ c2w
    c2w = c2w * mask
    c2w = c2w + 1e-8

    # Process Transform
    blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    c2w = c2w @ blender2opencv
    c2w = c2w.float()
    # c2w = torch.FloatTensor(pose)

    w, h = img_wh
    focal = 0.5 * 800 / np.tan(0.5 * math.pi * 30 / 180)  # original focal length
    focal *= img_wh[0] / 800  # modify focal length to match size self.img_wh

    directions = get_ray_directions(h, w, [focal,focal])  # (h, w, 3)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)

    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return torch.cat([rays_o, rays_d], 1)

def gaussian(x, sigma):
    return torch.exp(-0.5 * (x / sigma)**2)

def bilateral_filter(image, diameter, sigma_color, sigma_space, device='cuda'):
    # Create a grid for spatial Gaussian weights
    half_diameter = diameter // 2
    x = torch.arange(-half_diameter, half_diameter + 1, device=device).float()
    y = torch.arange(-half_diameter, half_diameter + 1, device=device).float()
    xv, yv = torch.meshgrid(x, y, indexing='ij')
    spatial_gaussian = gaussian(torch.sqrt(xv**2 + yv**2), sigma_space).view(1, 1, diameter, diameter, 1)

    # Unfold the image to get patches
    patches = F.unfold(image, kernel_size=diameter, padding=half_diameter)
    patches = patches.view(image.size(0), image.size(1), diameter, diameter, -1)

    if sigma_color > 0:
        # Compute the intensity difference directly
        center_pixel = image.view(image.size(0), image.size(1), 1, 1, -1)
        intensity_diff = patches - center_pixel
        # Compute the intensity Gaussian in-place
        intensity_gaussian = gaussian(intensity_diff, sigma_color)
        del intensity_diff
    else:
        intensity_gaussian = torch.ones_like(spatial_gaussian)

    # Combine spatial and intensity Gaussian weights
    weights = spatial_gaussian * intensity_gaussian
    del intensity_gaussian, spatial_gaussian

    # Normalize weights in-place
    weights_sum = weights.sum(dim=(2, 3), keepdim=True)
    weights /= weights_sum
    del weights_sum

    # Apply the filter and sum in-place
    output = (weights * patches).sum(dim=(2, 3))
    output = output.view(image.size(0), image.size(1), image.size(2), image.size(3))

    del patches, weights
    torch.cuda.empty_cache()

    return output

# @torch.no_grad()
def enhanced_evaluation(test_dataset, tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)

    idxs = list(range(0, test_dataset.all_rays.shape[0]))

    if args.idx is not None:
        print(f'Using view {args.idx} for evaluation')
        idxs = [args.idx]

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    
    print(f'Rendering:')
    for idx in tqdm(idxs):
        samples = test_dataset.all_rays[idx]
        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        for rep in range(1):
            start = time.time()
            rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                            ndc_ray=ndc_ray, white_bg = white_bg, device=device)
  
        rgb_map = rgb_map.clamp(0.0, 1.0)
        rgb_map = rgb_map.reshape(H, W, 3).cpu()

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idx].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
        
@torch.no_grad()
def evaluation(test_dataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    print(f"Rendering:")
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _ = renderer(rays, tensorf, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')

        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs
