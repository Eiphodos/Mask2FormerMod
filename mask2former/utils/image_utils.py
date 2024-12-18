import torch
import torch.nn
import torch.nn.functional as F
from einops import rearrange


def patches_to_images(patches, policy_code, grid_size):
    batch_size, dim_patch, patch_size, ps_times_num_patch = patches.size()
    num_patch = ps_times_num_patch // patch_size
    num_grid_h, num_grid_w = grid_size  # grid size is based on the original base patch size
    patches = rearrange(patches, 'b c hp (np wp) ->b np c hp wp', np=num_patch)
    num_total_grid = num_grid_h * num_grid_w

    scale_value = policy_code[:, :, 0]
    grid_coords = policy_code[:, :, 1:]
    max_scale = scale_value.max().item()

    # process patches that stay at the highest resolution
    scale_1_idx = scale_value == max_scale

    patch_scale_1 = patches[scale_1_idx]
    patch_scale_1 = rearrange(patch_scale_1, '(b np) c h w -> b np c h w', b=batch_size)
    grid_coord_1 = grid_coords[scale_1_idx].unsqueeze(1)
    grid_coord_1 = rearrange(grid_coord_1, '(b np) ng c -> b (np ng) c', b=batch_size)

    patches_uni_list = [patch_scale_1]
    grid_coord_uni_list = [grid_coord_1]

    for scale in range(0, max_scale):
        # process patches that are downsize by the factor of 2
        inv_scale = max_scale - scale
        scale_idx = scale_value == scale
        # transform 1*1 original grid coord to 2*2 (because it will be upsampled by the factor of 2)
        grid_coord = grid_coords[scale_idx].unsqueeze(1)  # coords are stacked across batch dim
        new_coords = torch.stack(torch.meshgrid(torch.arange(0, 2**inv_scale), torch.arange(0, 2**inv_scale), indexing='ij')).view(2,-1).permute(1,0).cuda()
        #new_coords = [grid_coord + nc for nc in new_coords]
        #grid_coord = torch.cat(new_coords, dim=1)
        grid_coord = grid_coord + new_coords
        grid_coord = rearrange(grid_coord, '(b np) ng c -> b (np ng) c', b=batch_size)  # create batch dim
        # find and enlarege the patches that are downsize before, and break it into 4 pieces
        patch_scale = patches[scale_idx]
        patch_scale = patch_scale.repeat(1, 1, 2 ** inv_scale, 2 ** inv_scale)
        #patch_scale = F.interpolate(patch_scale, scale_factor=2**inv_scale, mode='bilinear', align_corners=False, recompute_scale_factor=False)  # stacked across batch dim
        patch_scale = rearrange(patch_scale, 'b c (h1 h) (w1 w) -> b (h1 w1) c h w', h1=2**inv_scale, w1=2**inv_scale)
        patch_scale = rearrange(patch_scale, '(b np) ng c ps_h ps_w  -> b (np ng) c ps_h ps_w', b=batch_size)

        patches_uni_list.append(patch_scale)
        grid_coord_uni_list.append(grid_coord)



    # combine all the restored patches (of the same size and scale) together, total size should be 'num_total_grid'
    # even one shuffle the patches and grid value, it will be sorted later anyway
    patches_uni = torch.cat(patches_uni_list, dim=1)
    grid_coord_uni = torch.cat(grid_coord_uni_list, dim=1)

    # sort the patches according to grid universal positional value, different samples batch-level have offset values
    grid_uni_value = grid_coord_uni[:, :, 0] * num_grid_w + grid_coord_uni[:, :, 1]
    batch_offset = torch.linspace(0, batch_size - 1, batch_size).view(batch_size, 1).expand_as(grid_uni_value).cuda() * num_total_grid
    grid_sort_global = batch_offset + grid_uni_value
    grid_sort_global = grid_sort_global.view(-1)
    patch_uni_global = rearrange(patches_uni, 'b np c h w -> (b np) c h w')
    indices_global = torch.argsort(grid_sort_global)
    patch_uni_global = patch_uni_global[indices_global]

    patch_uni_global = rearrange(patch_uni_global, '(b np) c h w  -> b np c h w', b=batch_size)
    images = rearrange(patch_uni_global, 'b (hp wp) c h w -> b c (hp h) (wp w)', hp=num_grid_h, wp=num_grid_w)

    return images


def convert_1d_index_to_2d(one_dim_index, PS):
    x_coord = one_dim_index // PS
    y_coord = one_dim_index % PS
    two_dim_index = torch.stack([x_coord, y_coord])
    two_dim_index = two_dim_index.permute(1, 0)
    return two_dim_index

def convert_1d_patched_index_to_2d_org_index(one_dim_index, H, org_PS, scale, new_coord_PS):
    new_PS = org_PS // 2**scale
    new_H = H // new_PS
    x_coord = one_dim_index // new_H
    y_coord = one_dim_index % new_H
    org_x = x_coord * new_PS
    org_y = y_coord * new_PS
    new_x = org_x // new_coord_PS
    new_y = org_y // new_coord_PS
    two_dim_index = torch.stack([new_x, new_y])
    two_dim_index = two_dim_index.permute(1, 0)
    return two_dim_index

def convert_2d_index_to_1d(two_dim_index, PS):
    one_dim_index = two_dim_index[:, 0] * PS + two_dim_index[:, 1]
    return one_dim_index

def get_2d_coords_scale_from_h_w_ps(height, width, patch_size, scale):
    patches_coords = torch.meshgrid(torch.arange(0, height // patch_size), torch.arange(0, width // patch_size), indexing='ij')
    patches_coords = torch.stack([patches_coords[0], patches_coords[1]])
    patches_coords = patches_coords.permute(1, 2, 0)
    patches_coords = patches_coords.view(-1, 2)
    n_patches = patches_coords.shape[0]

    scale_lvl = torch.tensor([[scale]] * n_patches)
    patches_scale_coords = torch.cat([scale_lvl, patches_coords], dim=1)
    return patches_scale_coords

def get_1d_coords_scale_from_h_w_ps(height, width, patch_size, scale):
    n_patches = (height // patch_size) * (width // patch_size)
    patches_coords = torch.arange(n_patches).view(-1, 1)

    scale_lvl = torch.tensor([[scale]] * n_patches)
    patches_scale_coords = torch.cat([scale_lvl, patches_coords], dim=1)
    return patches_scale_coords.int()

def convert_scale_to_coords_in_full_res(one_dim_coords, patch_size, im_size):
    patch_size_2d = im_size // patch_size
    new_coords_all = []
    for coord in one_dim_coords:
        x_values = torch.arange((coord // patch_size_2d)*patch_size, (coord // patch_size_2d)*patch_size + patch_size)
        y_values = torch.arange((coord % patch_size_2d)*patch_size, (coord % patch_size_2d)*patch_size + patch_size)
        new_coords = torch.stack(torch.meshgrid(x_values, y_values, indexing='ij')).permute(1,2,0).view(-1, 2)
        new_coords_one_dim = convert_2d_index_to_1d(new_coords, im_size)
        new_coords_all.append(new_coords_one_dim)
    return torch.cat(new_coords_all, dim=0).long()


def create_oracle_labels(labels, patch_size): 
    max = F.max_pool2d(labels.float(), patch_size, stride=patch_size).to(torch.int32)
    min = F.max_pool2d(-labels.float(), patch_size, stride=patch_size).to(torch.int32)
    one_class = (max == -min)
    
    patch_groups_per_img = one_class.to(torch.uint8)
    
    return patch_groups_per_img.to(torch.uint8).squeeze(0)