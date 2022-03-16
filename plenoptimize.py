import os
import json
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image
import jax
import jax.numpy as jnp
import plenoxel
from jax.ops import index, index_add



def get_data(root, stage):
    all_c2w = []
    all_gt = []

    data_path = os.path.join(root, stage)
    data_json = os.path.join(root, 'transforms_' + stage + '.json')
    print('LOAD DATA', data_path)
    j = json.load(open(data_json, 'r'))

    for frame in tqdm(j['frames']):
        fpath = os.path.join(data_path, os.path.basename(frame['file_path']) + '.png')
        c2w = frame['transform_matrix']
        im_gt = imageio.imread(fpath).astype(np.float32) / 255.0
        im_gt = im_gt[..., :3] * im_gt[..., 3:] + (1.0 - im_gt[..., 3:])
        all_c2w.append(c2w)
        all_gt.append(im_gt)
    focal = 0.5 * all_gt[0].shape[1] / np.tan(0.5 * j['camera_angle_x'])
    all_gt = np.asarray(all_gt)
    all_c2w = np.asarray(all_c2w)
    return focal, all_c2w, all_gt


# low-pass filter the ground truth image so the effective resolution matches twice that of the grid
def lowpass(gt, resolution):
    if gt.ndim > 3:
        print(f'lowpass called on image with more than 3 dimensions; did you mean to use multi_lowpass?')
    H = gt.shape[0]
    W = gt.shape[1]
    im = Image.fromarray((np.squeeze(np.asarray(gt))*255).astype(np.uint8))
    im = im.resize(size=(resolution*2, resolution*2))
    im = im.resize(size=(H, W))
    return np.asarray(im) / 255.0


# low-pass filter a stack of images where the first dimension indexes over the images
def multi_lowpass(gt, resolution):
    if gt.ndim <= 3:
        print(f'multi_lowpass called on image with 3 or fewer dimensions; did you mean to use lowpass instead?')
    H = gt.shape[-3]
    W = gt.shape[-2]
    clean_gt = np.copy(gt)
    for i in range(len(gt)):
        im = Image.fromarray(np.squeeze(gt[i,...] * 255).astype(np.uint8))
        im = im.resize(size=(resolution*2, resolution*2))
        im = im.resize(size=(H, W))
        im = np.asarray(im) / 255.0
        clean_gt[i,...] = im
    return clean_gt


def get_loss(data_dict, c2w, gt, H, W, focal, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, occupancy_penalty, interpolation, nv):
    rays = plenoxel.get_rays(H, W, focal, c2w)
    rgb, disp, acc, weights, voxel_ids = plenoxel.render_rays(data_dict, rays, resolution, key, radius, harmonic_degree, jitter, uniform, interpolation, nv)
    mse = jnp.mean((rgb - lowpass(gt, resolution))**2)
    indices, data = data_dict
    loss = mse + occupancy_penalty * jnp.mean(jax.nn.relu(data[-1]))
    return loss


def get_loss_rays(data_dict, rays, gt, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, occupancy_penalty, interpolation, nv):
    rgb, disp, acc, weights, voxel_ids = plenoxel.render_rays(data_dict, rays, resolution, key, radius, harmonic_degree, jitter, uniform, interpolation, nv)
    mse = jnp.mean((rgb - gt)**2)
    indices, data = data_dict
    loss = mse + occupancy_penalty * jnp.mean(jax.nn.relu(data[-1]))
    return loss


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5, indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def render_pose_rays(data_dict, c2w, H, W, focal, resolution, radius, harmonic_degree, jitter, uniform, key, sh_dim, batch_size, interpolation, nv):
    rays_o, rays_d = get_rays_np(H, W, focal, c2w)
    rays_o = np.reshape(rays_o, [-1,3])
    rays_d = np.reshape(rays_d, [-1,3])
    rgbs = []
    disps = []
    for i in range(int(np.ceil(H*W/batch_size))):
        start = i*batch_size
        stop = min(H*W, (i+1)*batch_size)
        if jitter > 0:
            rgbi, dispi, acci, weightsi, voxel_idsi = jax.lax.stop_gradient(plenoxel.render_rays(data_dict, (rays_o[start:stop], rays_d[start:stop]), resolution, key[start:stop], radius, harmonic_degree, jitter, uniform, interpolation, nv))
        else:
            rgbi, dispi, acci, weightsi, voxel_idsi = jax.lax.stop_gradient(plenoxel.render_rays(data_dict, (rays_o[start:stop], rays_d[start:stop]), resolution, None, radius, harmonic_degree, jitter, uniform, interpolation, nv))
        rgbs.append(rgbi)
        disps.append(dispi)
    rgb = jnp.reshape(jnp.concatenate(rgbs, axis=0), (H, W, 3))
    disp = jnp.reshape(jnp.concatenate(disps, axis=0), (H, W))
    return rgb, disp, None, None


def run_test_step(i, data_dict, test_c2w, test_gt, H, W, focal, FLAGS, key, log_dir, name_appendage=''):
    print('Evaluating')
    sh_dim = (FLAGS.harmonic_degree + 1)**2
    tpsnr = 0.0
    for j, (c2w, gt) in tqdm(enumerate(zip(test_c2w, test_gt))):
        rgb, disp, _, _ = render_pose_rays(data_dict, c2w, H, W, focal, FLAGS.resolution, radius, FLAGS.harmonic_degree, FLAGS.jitter, FLAGS.uniform, key, sh_dim, FLAGS.physical_batch_size, FLAGS.interpolation, FLAGS.nv)
        mse = jnp.mean((rgb - gt)**2)
        psnr = -10.0 * np.log(mse) / np.log(10.0)
        tpsnr += psnr

        if FLAGS.render_interval > 0 and j % FLAGS.render_interval == 0:
            disp3 = jnp.concatenate((disp[...,jnp.newaxis], disp[...,jnp.newaxis], disp[...,jnp.newaxis]), axis=2)
            vis = jnp.concatenate((gt, rgb, disp3), axis=1)
            vis = np.asarray((vis * 255)).astype(np.uint8)
            imageio.imwrite(f"{log_dir}/{j:04}_{i:04}{name_appendage}.png", vis)
        del rgb, disp
    tpsnr /= n_test_imgs
    return tpsnr


def update_grid(old_grid, lr, grid_grad):
    return index_add(old_grid, index[...], -1 * lr * grid_grad)


def update_grids(old_grid, lrs, grid_grad):
    for i in range(len(old_grid)):
        old_grid[i] = index_add(old_grid[i], index[...], -1 * lrs[i] * grid_grad[i])
    return old_grid
