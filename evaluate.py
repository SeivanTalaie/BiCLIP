import argparse
import os
from engine.wrapper import LanGuideMedSegWrapper
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl  
from utils.new_dataset import ImageTextMaskDataset
import utils.config as config


def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation'
    )
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')

    parser.add_argument('--model_name',
                        type=str,
                        default=None,
                        help='Full import path for the model class, e.g., utils.githubUNET.github_UNET')

    parser.add_argument('--ckpt',
                        type=str,
                        default=None,
                        help='Full import path for the model class, e.g., utils.githubUNET.github_UNET')
    parser.add_argument('--dataset_name',
                        type=str,
                        default=None,
                        help='Full import path for the model class, e.g., utils.githubUNET.github_UNET')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    if args.model_name is not None:
        cfg['model_name'] = args.model_name
        cfg['ckpt'] = args.ckpt
        cfg['dataset_name'] = args.dataset_name
    return cfg


if __name__ == '__main__':

    import random
    import numpy as np

    seed = 43
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = get_parser()
    path = args.ckpt
    results = []
    for ckpt in os.listdir(path):
        if not ckpt.endswith('.ckpt'):
            continue
        print(ckpt)
        model = LanGuideMedSegWrapper(args)
        checkpoint = torch.load(os.path.join(path,ckpt), map_location='cuda', weights_only=False)["state_dict"]
        model.load_state_dict(checkpoint, strict=True)

        if args.dataset_name == 'kvasir':
            print('using dataset: kvasir-SEG')
            print('using dataset: kvasir-SEG')
            print('using dataset: kvasir-SEG')
            print('using dataset: kvasir-SEG')

            ds_test = ImageTextMaskDataset(
                tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
                prompt_type='p9',
                images_dir='/home/asosoft/Seivan/Kvasir_prepared/test/images',
                masks_dir='/home/asosoft/Seivan/Kvasir_prepared/test/masks',
                caps_file='/home/asosoft/Seivan/JSON_files/Kvasir/test.json'
            )


        elif args.dataset_name == 'covid':
            print('using dataset: QaTa-Cov19')
            print('using dataset: QaTa-Cov19')
            print('using dataset: QaTa-Cov19')
            print('using dataset: QaTa-Cov19')

            ds_test =  ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/home/asosoft/Seivan/QaTa_prepared/test/images',
            masks_dir='/home/asosoft/Seivan/QaTa_prepared/test/masks',
            caps_file='/home/asosoft/Seivan/JSON_files/QaTa/test.json'
        )


        elif args.dataset_name == 'mosmed':
            print('using dataset: MosMedData+')
            print('using dataset: MosMedData+')
            print('using dataset: MosMedData+')
            print('using dataset: MosMedData+')

            ds_test = ImageTextMaskDataset(
                tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
                prompt_type='p9',
                images_dir='/home/asosoft/Seivan/MosMedData+_prepared/test/images',
                masks_dir='/home/asosoft/Seivan/MosMedData+_prepared/test/masks',
                caps_file='/home/asosoft/Seivan/JSON_files/MosMedData+/test.json'
            )


        elif args.dataset_name == 'clinic':
            print('using dataset: CVC-ClinicDB')
            print('using dataset: CVC-ClinicDB')
            print('using dataset: CVC-ClinicDB')
            print('using dataset: CVC-ClinicDB')

            ds_test = ImageTextMaskDataset(
                tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
                prompt_type='p9',
                images_dir='/home/asosoft/Seivan/ClinicDB_prepared/test/images',
                masks_dir='/home/asosoft/Seivan/ClinicDB_prepared/test/masks',
                caps_file='/home/asosoft/Seivan/JSON_files/ClinicDB/test.json'
            )

        else:
            ValueError('No dataset')

    
        dl_test = DataLoader(ds_test, batch_size=args.valid_batch_size, shuffle=False, num_workers=4, drop_last=False)
        trainer = pl.Trainer(accelerator='gpu',devices=1)
        model.eval()
        result = trainer.test(model, dl_test)
        results.append(result[0])

    keys = results[0].keys()
    stats = {key: [] for key in keys}

    for d in results:
        for key in d:
            stats[key].append(d[key])

    print(stats.items())
    results = {key: {'mean': np.mean(values), 'std': np.std(values)} for key, values in stats.items()}

    for key, value in results.items():
        print(f"{key} - Mean: {value['mean']:.6f}, Std: {value['std']:.6f}")

    stats_file_path = os.path.join(args.ckpt, "stats.txt")
    with open(stats_file_path, "w") as f:
        for key, values in stats.items():
            f.write(f"{key}: {values}\n")

    results_file_path = os.path.join(args.ckpt, "results.txt")
    with open(results_file_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key} - Mean: {value['mean']:.6f}, Std: {value['std']:.6f}\n")

    print(f"Stats and results saved to {args.ckpt}")


# CUDA_VISIBLE_DEVICES=1 python evaluate.py --model_name utils.githubUNET.github_UNET_aug_text_gene.UNet --dataset_name mosmed --ckpt "/home/asosoft/Seivan/MosMedData+_Model/MosMed_Best.ckpt"
