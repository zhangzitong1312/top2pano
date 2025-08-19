import numpy as np
import torch,math

def quaternion_to_rotation_matrix(quaternion):
    """Convert quaternion (x, y, z, w) to a 3x3 rotation matrix."""
    x, y, z, w = [q.cpu().item() for q in quaternion]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    rotation_matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])

    return rotation_matrix

def get_original_coord(batch):
    '''
    pano_direction [X,Y,Z] x right,y up,z out
    '''
    W,H  = batch["ground_truth"].shape[2],batch["ground_truth"].shape[1]
    _y = np.repeat(np.array(range(W)).reshape(1,W), H, axis=0)
    _x = np.repeat(np.array(range(H)).reshape(1,H), W, axis=0).T
    _theta = (1 - 2 * (_x) / H) * np.pi / 2

    _phi = math.pi*( -0.5 - 2* (_y)/W )
    axis0 = (np.cos(_theta)*np.cos(_phi)).reshape(H, W, 1)
    axis1 = np.sin(_theta).reshape(H, W, 1) 
    axis2 = (-np.cos(_theta)*np.sin(_phi)).reshape(H, W, 1) 
    pano_direction = np.concatenate((axis0, axis1,axis2), axis=2)

    rotated_pano_directions = np.zeros((len(batch['rotation'][0]), H, W, 3))

    for i in range(len(batch['rotation'][0])):
        
        #Gibson
        if batch['dataset_name'][0] == 'gibson':
            quaternion = [batch['rotation'][0][i], batch['rotation'][2][i], batch['rotation'][1][i],
                      batch['rotation'][3][i]]  # (x, z, y, w)
            rotation_matrix = quaternion_to_rotation_matrix(quaternion)

        # Matterport
        if batch['dataset_name'][0] == 'matterport':
            rotation_matrix = torch.zeros(9)
            for j in range(9):
                rotation_matrix[j] = batch['rotation'][j][i]
            rotation_matrix = rotation_matrix.cpu().numpy()
            rotation_matrix = rotation_matrix.reshape(3, 3)
            rotation_matrix[[0, 1, 2]] = rotation_matrix[[0, 2, 1]]

        rotated_pano_directions[i] = np.einsum('ij,hwj->hwi', rotation_matrix, pano_direction)

    return rotated_pano_directions


def render(batch,feature,voxel,pano_direction,PE=None):

    sat_W,sat_H = batch['jpg'].shape[1], batch['jpg'].shape[2]
    BS = feature.shape[0]
    sample_number = 150
    max_height = 3
    render_distance = 0.8

    origin_height = (batch['location'][2]).cpu()
    pixel_resolution = (1 / batch['pixel_scale']).cpu()
    realworld_scale = (pixel_resolution * sat_W).cpu()

    for i in range(BS):
        if batch['location'][2][i] > 6:
            batch['location'][2][i] -= 6
        elif batch['location'][2][i] > 3:
            batch['location'][2][i] -= 3
        elif batch['location'][2][i] < 0:
            batch['location'][2][i] += 3


    sample_total_length_rgb = np.ones(BS) * render_distance
    sample_total_length_depth = np.zeros(BS)
    for i in range(BS):
        sample_total_length_depth[i] = (int((np.sqrt((realworld_scale[i] / 2) ** 2 + (realworld_scale[i] / 2) ** 2 + (2) ** 2))/pixel_resolution[i]))/(sat_W/2)

    origin_z = torch.ones([BS,1])*(-1+(origin_height/(realworld_scale/2))) ### -1 is the loweast position in regular cooridinate
    origin_z = origin_z[0].unsqueeze(-1)


    screen_location = batch['screen_location']
    origin_H_W = {}
    for i in range(BS):
        origin_H_W[i] = (256-screen_location[1][i].cpu().item())/256, (screen_location[0][i].cpu().item()-256)/256

    origin_H = torch.zeros(BS,1)
    origin_w = torch.zeros(BS,1)
    for i in range(BS):
        origin_H[i] = torch.ones(1) * origin_H_W[i][0]
        origin_w[i] = torch.ones(1) * origin_H_W[i][1]

    origin = torch.cat([origin_w,origin_z,origin_H],dim=1).to('cuda')[:,None,None,:]  ## w,z,h, samiliar to NERF coordinate definition

    sample_len = torch.zeros(BS, sample_number, device='cuda')
    sample_len_depth = torch.zeros(BS, sample_number, device='cuda')
    for i in range(BS):
        sample_len[i] = ((torch.arange(sample_number)+1)*(sample_total_length_rgb[i]/sample_number)).to('cuda')
        sample_len_depth[i] = ((torch.arange(sample_number)+1)*(sample_total_length_depth[i]/sample_number)).to('cuda')

    origin = origin[...,None]
    pano_direction = pano_direction[...,None]
    depth = sample_len.view(BS,1, 1, 1, -1).to('cuda')
    sample_point = origin + pano_direction * depth
    
    depth_depth = sample_len_depth.view(BS,1, 1, 1, -1).to('cuda')
    sample_point_depth = origin + pano_direction * depth_depth

    N = voxel.size(1)
    voxel_low = -1
    voxel_max = -1 + max_height/(realworld_scale/2)
    grid = sample_point.permute(0,4,1,2,3)[...,[0,2,1]]
    grid_depth = sample_point_depth.permute(0,4,1,2,3)[...,[0,2,1]]

    for i in range(BS):
        grid[...,2][i] = ((grid[...,2][i]-voxel_low)/(voxel_max[i]-voxel_low))*2-1
        grid_depth[...,2][i] = ((grid_depth[...,2][i]-voxel_low)/(voxel_max[i]-voxel_low))*2-1
    grid = grid.float()
    grid_depth = grid_depth.float()

    feature = feature.permute(0,3,1,2)
    color_input = feature.unsqueeze(2).repeat(1, 1, N, 1, 1)
   

    alpha_grid = torch.nn.functional.grid_sample(voxel.unsqueeze(1), grid)

    color_grid = torch.nn.functional.grid_sample(color_input, grid)

    alpha_grid_depth = torch.nn.functional.grid_sample(voxel.unsqueeze(1), grid_depth)

    depth_sample = depth.permute(0,1,2,4,3).view(1,-1,sample_number,1)
    depth_sample = depth_sample.permute(1,0,2,3)
    depth_sample_depth = depth_depth.permute(0,1,2,4,3).view(1,-1,sample_number,1)
    depth_sample_depth = depth_sample_depth.permute(1,0,2,3)

    feature_size = color_grid.size(1)
    color_grid = color_grid.permute(0,3,4,2,1).view(BS,-1,sample_number,feature_size)
    alpha_grid = alpha_grid.permute(0,3,4,2,1).view(BS,-1,sample_number)
    alpha_grid_depth = alpha_grid_depth.permute(0,3,4,2,1).view(BS,-1,sample_number)
    intv = sample_total_length_rgb/ sample_number
    intv = torch.from_numpy(intv).to('cuda')

    intv_depth = sample_total_length_depth / sample_number
    intv_depth = torch.from_numpy(intv_depth).to('cuda')

    output = composite_rgb(batch, BS,rgb_samples=color_grid,density_samples=alpha_grid,intv = intv)
    output['depth'] = compositie_depth(batch, BS, density_samples=alpha_grid_depth, depth_samples=depth_sample_depth, intv=intv_depth)
    output['voxel']  = voxel
    return output

def composite(batch,BS, rgb_samples,density_samples,depth_samples,intv):
    sigma_delta = torch.zeros(BS,density_samples.shape[1], density_samples.shape[2], device=density_samples.device)
    for i in range(BS):
        sigma_delta[i] = density_samples[i]*float(intv[i].item())
    alpha = 1-(-sigma_delta).exp_()
    T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)) .exp_()
    prob = (T*alpha)[...,None]
    depth = (depth_samples*prob).sum(dim=2)
    rgb = (rgb_samples*prob).sum(dim=2)
    depth = depth.permute(0,2,1).view(depth.size(0),-1,batch['ground_truth'].shape[2],batch['ground_truth'].shape[1])
    rgb = rgb.permute(0,2,1).view(rgb.size(0),-1,batch['ground_truth'].shape[2],batch['ground_truth'].shape[1])
    rgb = rgb.permute(0,3,2,1)
    epsilon = 1e-8
    for i in range(rgb.shape[0]):
        rgb[i] = (rgb[i] - rgb[i].min()) / (rgb[i].max() - rgb[i].min())
        depth[i] = (depth[i] - depth[i].min()) / (depth[i].max() - depth[i].min()+epsilon)
    return {'rgb': rgb, 'depth': depth}

def composite_rgb(batch,BS,rgb_samples,density_samples,intv):
    sigma_delta = torch.zeros(BS,density_samples.shape[1], density_samples.shape[2], device=density_samples.device)
    for i in range(BS):
        sigma_delta[i] = density_samples[i]*float(intv[i].item())
    alpha = 1-(-sigma_delta).exp_()
    T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)) .exp_()
    prob = (T*alpha)[...,None]
    
    rgb = (rgb_samples*prob).sum(dim=2)
    rgb = rgb.permute(0,2,1).view(rgb.size(0),-1,batch['ground_truth'].shape[2],batch['ground_truth'].shape[1])
    rgb = rgb.permute(0,3,2,1)
    
    for i in range(rgb.shape[0]):
        rgb[i] = (rgb[i] - rgb[i].min()) / (rgb[i].max() - rgb[i].min())
    return {'rgb': rgb}


def compositie_depth(batch,BS, density_samples,depth_samples,intv):
    sigma_delta = torch.zeros(BS,density_samples.shape[1], density_samples.shape[2], device=density_samples.device)
    for i in range(BS):
        sigma_delta[i] = density_samples[i]*float(intv[i].item())
    alpha = 1-(-sigma_delta).exp_()
    T = (-torch.cat([torch.zeros_like(sigma_delta[...,:1]),sigma_delta[...,:-1]],dim=2).cumsum(dim=2)) .exp_()
    prob = (T*alpha)[...,None]
    depth = (depth_samples*prob).sum(dim=2)
    depth = depth.permute(0,2,1).view(depth.size(0),-1,batch['ground_truth'].shape[2],batch['ground_truth'].shape[1])
    epsilon = 1e-8
    for i in range(depth.shape[0]):
        depth[i] = (depth[i] - depth[i].min()) / (depth[i].max() - depth[i].min()+epsilon)
    return depth


def density_map(batch):

    pano_direction = get_original_coord(batch)
    pano_direction = torch.from_numpy(pano_direction).to('cuda')
    for i in range(batch['voxel'].shape[0]):
        batch['voxel'][i] = (batch['voxel'][i] - batch['voxel'][i].min()) / (batch['voxel'][i].max() - batch['voxel'][i].min())
    batch['voxel'] = batch['voxel'] * 10.0

    # solid wall
    for i in range(batch['voxel'].shape[0]):
        mask = batch['black_mask'][i].permute(2,0,1)
        mask = mask.expand(*batch['voxel'][i].shape)
        batch['voxel'][i][mask] = 1000

    # solid floor 
    batch['voxel'] = torch.cat([torch.ones(batch['voxel'].size(0),1,batch['voxel'].size(2),batch['voxel'].size(3),device='cuda')*1000,batch['voxel']],1)
    batch['voxel'] = torch.cat([torch.ones(batch['voxel'].size(0),1,batch['voxel'].size(2),batch['voxel'].size(3),device='cuda')*1000,batch['voxel']],1)
    
    output = render(batch, batch['jpg'], batch['voxel'], pano_direction)
    return output['rgb'], output['depth']





