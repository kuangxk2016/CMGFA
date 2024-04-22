# CMGFA

The complete code will be updated after the article is accepted.

## Results

### BEV Map Segmentation (on nuScenes validation)

|   Model   | Modality |  mIoU  | 
|:---------:|:--------:|:------:|
| BEVFusion |  C+L+R   | 0.6207 |
|    CMFA    |    C+L+R     | 0.6462  |
|  CMGFA   |    C+L+R     | 0.6959  |

## Usage

### Prerequisites

The code is built with following libraries:

- Python >= 3.8, \<3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0 (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, \<= 1.10.2
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.20.0
- [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)

After installing these dependencies, please run this command to install the codebase:

```bash
python setup.py develop
```

### Data Preparation

#### nuScenes

Please follow the instructions from [here](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/datasets/nuscenes_det.md) to download and preprocess the nuScenes dataset. Please remember to download both detection dataset and the map extension (for BEV map segmentation). After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── nuscenes_database
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   ├── nuscenes_infos_test.pkl
│   │   ├── nuscenes_dbinfos_train.pkl
│   │   ├── nuscenes_radar
│   │   │   ├── nuscenes_radar_infos_train_radar.pkl
│   │   │   ├── nuscenes_radar_infos_val_radar.pkl
```

### Evaluation

We also provide instructions for evaluating our models: 

```bash
./tools/download_pretrained.sh
```

### Training

We provide instructions to reproduce our results on nuScenes.

For example, if you want to train the C+R+L for our model, please run:

```bash
torchpack dist-run -np 4 python tools/train.py configs/groupmix-qkv.yaml
```
Note:
Train on the basis of [XXX.pth], please add --load_from [checkpoint name].pth.

Restore progress training to [XXX.pth], please add --resume_from [checkpoint name].pth
### Testing/Val

```bash
torchpack dist-run -np 4 python tools/test.py [config file path] pretrained/[checkpoint name].pth --eval map
```

Note: We will then update all checkpoint files



## Acknowledgements

----

## Citation

---

