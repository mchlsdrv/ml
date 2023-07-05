import os
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights
from torchvision.utils import save_image
import tqdm


class VGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:29]

    def forward(self, x):
        feats = []

        for lyr_n, lyr in enumerate(self.model):
            x = lyr(x)

            if str(lyr_n) in self.chosen_features:
                feats.append(x)

        return feats


def load_image(image_file: str or pathlib.Path, device: torch.device):
    img = Image.open(str(image_file))
    img = loader(img).unsqueeze(0)
    return img.to(device)


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on: {dev}')
img_h = 630
img_w = 1200

loader = transforms.Compose(
    [
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
    ]
)


IMGS_DIR = pathlib.Path('C:/Users/Michael/Pictures')
OUTPUT_DIR = pathlib.Path('C:/Users/Michael/Desktop/dev/ml/dl/cnns') / 'outputs/style_transfer'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# img_orig = load_image(IMGS_DIR / 'lion.jpg', device=dev)
img_name = 'lion'
img_orig = load_image(IMGS_DIR / f'{img_name}.jpg', device=dev)
styl_name = 'picasso-2'
# styl_name = 'van-gogh-1'
# styl_name = 'Stalin-1'
img_styl = load_image(IMGS_DIR / f'{styl_name}.jpg', device=dev)

img_gen = img_orig.clone().requires_grad_(True)

# Hyperparameters
total_steps = 15000
learning_rate = 0.001
alpha = 1
beta = 0.1
opt = optim.Adam([img_gen], lr=learning_rate)

mdl = VGG().to(device=dev).eval()

for step in tqdm.tqdm(range(total_steps)):
    feats_gen = mdl(img_gen)
    feats_orig = mdl(img_orig)
    feats_styl = mdl(img_styl)

    loss_styl = loss_orig = 0

    for feat_gen, feat_orig, feat_styl in zip(feats_gen, feats_orig, feats_styl):
        btch_sz, ch, h, w = feat_gen.shape
        loss_orig += torch.mean((feat_gen - feat_orig) ** 2)

        # Compute the Gram Matrix
        G = feat_gen.view(ch, h * w).mm(
            feat_gen.view(ch, h * w).t()
        )

        A = feat_styl.view(ch, h * w).mm(
            feat_styl.view(ch, h * w).t()
        )

        loss_styl += torch.mean((G - A) ** 2)

    loss_total = alpha * loss_orig + beta * loss_styl

    opt.zero_grad()

    loss_total.backward()

    opt.step()

    if step % 200 == 0:
        print(f'> Total loss for step {step}: {loss_total:.2f}')

        save_dir = OUTPUT_DIR / f'{img_name}/{styl_name}_style'
        os.makedirs(save_dir, exist_ok=True)

        save_image(img_gen, str(save_dir / f'gen_img_{step}.png'))

