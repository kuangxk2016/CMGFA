from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
import math

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

__all__ = ["GMMFA"]


@FUSIONMODELS.register_module()
class GMMFA(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        fuser_cam: Dict[str, Any],
        fuser_MFA: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("inf_camera") is not None:
            self.encoders["inf_camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["inf_camera"]["backbone"]),
                    "neck": build_neck(encoders["inf_camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["inf_camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if encoders.get("radar") is not None:
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

        if fuser_cam is not None:
            self.fuser_cam = build_fuser(fuser_cam)
        else:
            self.fuser_cam = None
        if fuser_MFA is not None:
            self.fuser_MFA = build_fuser(fuser_MFA)
        else:
            self.fuser_MFA = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']


        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        sensor,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders[sensor]["backbone"](x)
        x = self.encoders[sensor]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders[sensor]["vtransform"](
            x,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss, 
            gt_depths=gt_depths,
        )
        return x
    
    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x


    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if sensor == 'lidar':
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        features_cam = []
        auxiliary_losses = {}
        feat_cam = None
        feat_l = None
        feat_r = None
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    sensor,
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
                    metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
                feature = torch.tanh(feature)
                features_cam.append(feature)
            elif sensor == "inf_camera":
                feature = self.extract_camera_features(
                    sensor,
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
                    metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
                features_cam.append(feature)
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)
                features.append(feature)
                feat_l = feature
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
                features.append(feature)
                feat_r = feature
            else:
                raise ValueError(f"unsupported sensor: {sensor}")



        if not self.training:
            # avoid OOM
            features = features[::-1]
            features_cam = features_cam[::-1]

        if self.fuser_cam is not None:
            feat_temp = self.fuser_cam(features_cam)
            features.append(feat_temp)
            feat_cam = feat_temp
        if self.fuser_MFA is not None:
            x = self.fuser_MFA(feat_cam, feat_l, feat_r) #

        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                    #loss_imu = head(x_imu. gt_masks_bev[drivable_area])
                    #losses = losses+loss)imu
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    #logits_imu = head_imu(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

