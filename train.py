import gc
import torch
from torch.utils.data import DataLoader
from utils.new_dataset import ImageTextMaskDataset
import utils.config as config
from torch.optim import lr_scheduler
from engine.wrapper import LanGuideMedSegWrapper
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torchmetrics import Accuracy, Dice
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.multiprocessing
import argparse
import warnings

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
torch.multiprocessing.set_sharing_strategy('file_system')
def get_parser():
    parser = argparse.ArgumentParser(
        description='Robust Medical Image Segmentation'
    )
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')

    parser.add_argument('--model_name',
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
        cfg['dataset_name'] = args.dataset_name
    return cfg


if __name__ == '__main__':
    args = get_parser()
    print("cuda:", torch.cuda.is_available())
    save_path = "/home/asosoft/Seivan/MosMedData+_Model/"

    if args.dataset_name == 'mosmed':
        print('using dataset: MosMedData+')
        print('using dataset: MosMedData+')
        print('using dataset: MosMedData+')
        print('using dataset: MosMedData+')

        ds_train = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/home/asosoft/Seivan/MosMedData+_prepared/train/images',
            masks_dir='/home/asosoft/Seivan/MosMedData+_prepared/train/masks',
            caps_file='/home/asosoft/Seivan/JSON_files/MosMedData+/train.json',
        )
        ds_valid = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/home/asosoft/Seivan/MosMedData+_prepared/val/images',
            masks_dir='/home/asosoft/Seivan/MosMedData+_prepared/val/masks',
            caps_file='/home/asosoft/Seivan/JSON_files/MosMedData+/val.json',
        )


    elif args.dataset_name == 'kvasir':
        print('using dataset: kvasir')
        print('using dataset: kvasir')
        print('using dataset: kvasir')
        print('using dataset: kvasir')

        ds_train = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/home/asosoft/Seivan/Kvasir_prepared/train/images',
            masks_dir='/home/asosoft/Seivan/Kvasir_prepared/train/masks',
            caps_file='/home/asosoft/Seivan/JSON_files/Kvasir/train.json',
        )
        ds_valid = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/home/asosoft/Seivan/Kvasir_prepared/val/images',
            masks_dir='/home/asosoft/Seivan/Kvasir_prepared/val/images',
            caps_file='/home/asosoft/Seivan/JSON_files/Kvasir/val.json',
        )


    elif args.dataset_name == 'covid':
        print('using dataset: QaTa-Covid19')
        print('using dataset: QaTa-Covid19')
        print('using dataset: QaTa-Covid19')
        print('using dataset: QaTa-Covid19')

        ds_train = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/home/asosoft/Seivan/QaTa_prepared/train/images',
            masks_dir='/home/asosoft/Seivan/QaTa_prepared/train/masks',
            caps_file='/home/asosoft/Seivan/JSON_files/QaTa/train.json',
        )
        ds_valid = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/home/asosoft/Seivan/QaTa_prepared/val/images',
            masks_dir='/home/asosoft/Seivan/QaTa_prepared/val/masks',
            caps_file='/home/asosoft/Seivan/JSON_files/QaTa/val.json',
        )


    elif args.dataset_name == 'clinic':
        print('using dataset: CVC-ClinicDB')
        print('using dataset: CVC-ClinicDB')
        print('using dataset: CVC-ClinicDB')
        print('using dataset: CVC-ClinicDB')

        ds_train = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/home/asosoft/Seivan/ClinicDB_prepared/train/images',
            masks_dir='/home/asosoft/Seivan/ClinicDB_prepared/train/masks',
            caps_file='/home/asosoft/Seivan/JSON_files/ClinicDB/train.json',
        )
        ds_valid = ImageTextMaskDataset(
            tokenizer_type="microsoft/BiomedVLP-CXR-BERT-specialized",
            prompt_type='p9',
            images_dir='/home/asosoft/Seivan/ClinicDB_prepared/val/images',
            masks_dir='/home/asosoft/Seivan/ClinicDB_prepared/val/masks',
            caps_file='/home/asosoft/Seivan/JSON_files/ClinicDB/val.json',
        )

    else:
        ValueError('No dataset')

    dl_train = DataLoader(ds_train, batch_size=1, shuffle=True, num_workers=32, persistent_workers=True)
    dl_valid = DataLoader(ds_valid, batch_size=1, shuffle=False, num_workers=32, persistent_workers=True)

    for i in range(5):  # Repeat the experiment 5 times
        print(f"Starting experiment {i+1}")
        pl.seed_everything(44+i)
        model = LanGuideMedSegWrapper(args)

        # Setting checkpoint and early stopping callbacks
        model_ckpt = ModelCheckpoint(
            dirpath= save_path + str(args.dataset_name),
            filename=f"{args.model_name.split('.')[-1]}_EXP_{i+1}_{{epoch}}_{{val_loss:.4f}}_{{val_dice:.4f}}_{{val_MIoU:.4f}}",
            monitor='val_dice',
            save_top_k=1,
            mode='max',
            verbose=True)

        logger = TensorBoardLogger(
            save_dir=save_path + "logs",
            name=f"BiCLIP_{args.dataset_name}",
            version=f"EXP{i+1}")

        # Initialize trainer
        trainer = pl.Trainer(
            logger=logger,
            min_epochs=150,
            max_epochs=150,
            accelerator='gpu',
            devices=[1],
            # strategy="ddp",
            callbacks=[model_ckpt],
            enable_progress_bar=True)

        # Start training
        print('Start training')
        trainer.fit(model, dl_train, dl_valid)
        print('Done training')

        del model
        torch.cuda.empty_cache()
        gc.collect()


# CUDA_VISIBLE_DEVICES=1 python train.py --model_name utils.githubUNET.github_UNET_aug_text_gene.UNet --dataset_name mosmed