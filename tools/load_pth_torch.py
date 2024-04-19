import torch
import copy
import argparse
from torchpack.utils.config import configs
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
from mmcv import Config

def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--checkpoint", type=str, default=)

    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model_dir = args.checkpoint
    print(f'Loaded weights from \n {model_dir}')
    model.eval()
    device = torch.device('cuda:4')
    model.to(device)

    total_params = 0
    for name, child in model.named_children():
        params = count_params(child)
        total_params += params
        print("{} modal parameters：{}".format(name, params))
    print("all parameters：{}个".format(total_params))

if __name__ == '__main__':
    main()
