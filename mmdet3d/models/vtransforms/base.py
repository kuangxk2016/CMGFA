from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import bev_pool

__all__ = ["BaseTransform", "BaseDepthTransform"]

def boolmask2idx(mask):
    # A utility function, workaround for ONNX not supporting 'nonzero'
    return torch.nonzero(mask).squeeze(1).tolist()

def gen_dx_bx(xbound, ybound, zbound): #计算的都是什么
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]) #dx也就是row[2],是xyzbound的第三个元素，也就是步长
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]]) #栅格左边界
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    ) #三个维度的栅格数
    return dx, bx, nx


'''
LSS算法
1）生成视锥，并根据相机内外参将视锥中的点投影到ego坐标系
2）对环视图像完成特征的提取，并构建图像特征点云
3）利用变换后的ego坐标系的点与图像特征点云利用Voxel Pooling构建BEV特征
'''
class BaseTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        use_points='lidar', 
        depth_input='scalar',
        height_expand=True,
        add_depth_features=True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        self.use_points = use_points
        assert use_points in ['radar', 'lidar']
        self.depth_input=depth_input
        assert depth_input in ['scalar', 'one-hot']
        self.height_expand = height_expand
        self.add_depth_features = add_depth_features

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self): #生成视锥网格
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)#创建从1到60的等差数列，步长为0.5
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        ) #输出大小为(60-1)/0.5,fH,fW
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float) #创建大小从0到iw-1，均匀分布，个数为fW的tensor
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )
        #堆积形成网格坐标，frustum[i,j,k,0]就是(i,j)位置，深度为K的像素的宽度方向上的栅格坐标
        frustum = torch.stack((xs, ys, ds), -1) #fw,fh,D ,3
        return nn.Parameter(frustum, requires_grad=False)

    @force_fp32()
    def get_geometry( #锥点由图像坐标系向ego坐标系进行坐标转换，涉及相机内外参数
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape #N是相机个数

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对象素的变化
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins)) #反归一化
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3) #（b,N,depth,h,w,3）
        #上述Points的物理含义是每个batch中的每个环视相机图像特征点，其在不同深度下位置对应ego坐标系下的坐标

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C) #特征点云展平

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()#ego下的空间坐标转到体素坐标（计算栅格坐标并取整）
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1) #(BNDHW,4)

        # filter out points that are outside box
        #比如x:0~199, y:0~199, z:0
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    @force_fp32()
    def forward(
        self,
        img,
        points,
        radar, 
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,  #新加的
        **kwargs,
    ):
        rots = camera2ego[..., :3, :3]  #相机坐标系->车辆坐标系的旋转矩阵 rots=(batch,N,3,3)
        trans = camera2ego[..., :3, 3]  #相机坐标系->车辆坐标系的平移矩阵 rots=(batch,N,3)
        intrins = camera_intrinsics[..., :3, :3] #相机内参(batch,N,3,3)
        post_rots = img_aug_matrix[..., :3, :3] #由图像增强引起的旋转矩阵
        post_trans = img_aug_matrix[..., :3, 3] #由图像增强引起的平移矩阵
        lidar2ego_rots = lidar2ego[..., :3, :3] #激光雷达->车辆坐标系
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3] #相机->激光雷达
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )
        mats_dict = {
            'intrin_mats': camera_intrinsics, 
            'ida_mats': img_aug_matrix, 
            'bda_mat': lidar_aug_matrix,
            'sensor2ego_mats': camera2ego, 
        }
        x = self.get_cam_feats(img)#, mats_dict

        use_depth = False
        if type(x) == tuple:
            x, depth = x 
            use_depth = True
        
        x = self.bev_pool(geom, x) #geom（b,N,depth,h,w,3）, x (b,N,D fh,fw,C)

        if use_depth:
            return x, depth 
        else:
            return x



class BaseDepthTransform(BaseTransform):
    @force_fp32()
    def forward(
        self,
        img,
        points,
        radar, 
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        rots = sensor2ego[..., :3, :3]
        trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2ego_rots = lidar2ego[..., :3, :3]
        lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        if self.use_points == 'radar':
            points = radar

        if self.height_expand:
            for b in range(len(points)):
                points_repeated = points[b].repeat_interleave(8, dim=0)
                points_repeated[:, 2] = torch.arange(0.25, 2.25, 0.25).repeat(points[b].shape[0])
                points[b] = points_repeated

        batch_size = len(points)
        depth_in_channels = 1 if self.depth_input=='scalar' else self.D
        if self.add_depth_features:
            depth_in_channels += points[0].shape[1]

        depth = torch.zeros(batch_size, img.shape[1], depth_in_channels, *self.image_size, device=points[0].device)


        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]

                if self.depth_input == 'scalar':
                    depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
                elif self.depth_input == 'one-hot':
                    # Clamp depths that are too big to D
                    # These can arise when the point range filter is different from the dbound. 
                    masked_dist = torch.clamp(masked_dist, max=self.D-1)
                    depth[b, c, masked_dist.long(), masked_coords[:, 0], masked_coords[:, 1]] = 1.0

                if self.add_depth_features:
                    depth[b, c, -points[b].shape[-1]:, masked_coords[:, 0], masked_coords[:, 1]] = points[b][boolmask2idx(on_img[c])].transpose(0,1)

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        mats_dict = {
            'intrin_mats': intrins, 
            'ida_mats': img_aug_matrix, 
            'bda_mat': lidar_aug_matrix,
            'sensor2ego_mats': sensor2ego, 
        }
        x = self.get_cam_feats(img, depth, mats_dict)

        use_depth = False
        if type(x) == tuple:
            x, depth = x 
            use_depth = True
        
        x = self.bev_pool(geom, x)

        if use_depth:
            return x, depth 
        else:
            return x

