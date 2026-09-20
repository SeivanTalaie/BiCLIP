########################### Libraries ############################
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from kornia.augmentation import AugmentationSequential, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine, Normalize, RandomContrast, RandomBrightness, RandomGaussianNoise, RandomGamma, RandomMotionBlur, 


########################### BERT model ############################
class BERTModel(nn.Module):
    def __init__(self, bert_type = 'microsoft/BiomedVLP-CXR-BERT-specialized', project_dim = 784):
        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),
            nn.GELU(),
            nn.Linear(project_dim, project_dim)
        )
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) 
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
        embed = self.project_head(embed)

        return {'feature':output['hidden_states'],'project':embed}


######################## Image2Text Head ########################
class ImageToTextHead(nn.Module):
    def __init__(self, img_channels=3, text_dim=784):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, text_dim)

    def forward(self, x):
        x = self.encoder(x)        # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)  # (B, 64)
        return self.fc(x)          # [B, text_dim]


##################### Image feature extractor ##################### 
########## MLP
class ImageFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feat_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, feat_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))

    def forward(self, x):
        x = self.conv(x)  # [B, feat_dim, 1, 1]
        return x.view(x.size(0), -1)  # (B, feat_dim)
    

############################ Text Refiner ##########################
######## MLP
class TextRefiner(nn.Module):
    def __init__(self, text_dim=784, img_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim + img_dim, text_dim),
            nn.LayerNorm(text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim))

    def forward(self, text_embed, img_embed):
        x = torch.cat([text_embed, img_embed], dim=1)  # [B, 1040]
        delta = self.mlp(x)  # [B, 784]
        return text_embed + delta  # [B, 784]

 
########################### UNet model ############################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels = 6, n_classes = 1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        self.text_encoder = BERTModel()

        #### MLP 
        self.image_feature_extractor = ImageFeatureExtractor(in_channels=3)
        self.text_refiner = TextRefiner(text_dim=784, img_dim=256)
        
        self.image_to_text = ImageToTextHead(img_channels=3, text_dim=784)

        self.geom_aug = AugmentationSequential(
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomAffine(
                degrees=20,
                translate=(0.1, 0.1),
                scale=(0.8, 1.2),
                p=0.7
            ),
            data_keys=["input", "mask"],
        )

        self.photo_ref = AugmentationSequential(
            Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
            data_keys=["input"],
        )

        self.norm_pseudo = AugmentationSequential(
            Normalize(
                mean=torch.tensor([0.5, 0.5, 0.5]),
                std=torch.tensor([0.25, 0.25, 0.25]),
            ),
            data_keys=["input"],
        )

        self.photo_strong = AugmentationSequential(
            RandomGamma((0.9, 1.1)),
            RandomGaussianNoise(0.0, 0.01, p=0.5),
            RandomBrightness(brightness=0.3, p=0.5),
            RandomContrast(contrast=0.3, p=0.5),
            RandomMotionBlur(5, 20.0, 0.5, p=0.2),
            Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
            data_keys=["input"],
        )

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),  
            nn.Sigmoid()
        )
        
        self.IAC_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.LayerNorm(256),
        )

        print('======================================')
        print('github_UNET_aug_text generate')
        print('======================================')
  

    def decoder_features(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        d_feats = self.up4(x, x1)
        return d_feats


    def forward(self, data):
        x, _text, gt, gt_generate, img_name, real_text = data
        text = self.text_encoder(_text['input_ids'],_text['attention_mask'])

        text_embed = text['project']    # [B, 784]

        img_feat = self.image_feature_extractor(x)  # [B, 256]
        refined_text = self.text_refiner(text_embed, img_feat)  # [B, 784]

        # ---- Text → image
        batch, _ = refined_text.shape
        reshaped_B1 = refined_text.reshape(batch,1,28,28)
        pseudo_img = self.generator(reshaped_B1)  # [B, 3, 224, 224]

        # ---- Cycle consistency
        reconstructed_text = self.image_to_text(pseudo_img) # [B, 784]

        x = torch.cat([x, pseudo_img],dim = 1)

        if self.training:
            x_geo, gt = self.geom_aug(x, gt) # [B, 6, 224, 224], [B, 1, 224, 224]
            real_geo = x_geo[:, :3]
            pseudo_geo = x_geo[:, 3:]

            real_ref = self.photo_ref(real_geo)
            real_strong = self.photo_strong(real_geo)
            pseudo_norm = self.norm_pseudo(pseudo_geo)

            x_ref = torch.cat([real_ref, pseudo_norm], dim=1)
            x_aug = torch.cat([real_strong, pseudo_norm], dim=1)

        else:
            real = x[:, :3]
            pseudo = x[:, 3:]

            real_ref = self.photo_ref(real)
            pseudo_norm = self.norm_pseudo(pseudo)

            x_ref = torch.cat([real_ref, pseudo_norm], dim=1)
            x_aug = x_ref  # makes IAC ≈ 0 during val/test (more stable losses)
            
        gt = (gt > 0.5).int()

        # features
        feat_w = self.decoder_features(x_ref)  # [32, 64, 224, 224]
        feat_s = self.decoder_features(x_aug)  # [32, 64, 224, 224] 
        feat_w_proj = self.IAC_proj(feat_w)  # [B, 256]
        feat_s_proj = self.IAC_proj(feat_s)  # [B, 256]
        logits = self.outc(feat_w)  # [32, 1, 224, 224]

        return torch.sigmoid(logits), gt, pseudo_img, gt_generate, feat_w_proj, feat_s_proj, text_embed, reconstructed_text

