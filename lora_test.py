import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lora_dataset import MyDataset
from top_down_dataset_test import MyDataset

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch.nn as nn
import numpy as np

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4, alpha=4):
        super().__init__()
        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)
        nn.init.normal_(self.lora_A.weight, std=1 / rank)
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / rank

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scale

class LoRALinear(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=4):
        super().__init__()
        self.linear = original_layer
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank=rank,
            alpha=alpha
        )

        self.weight = self.linear.weight
        self.bias = self.linear.bias if self.linear.bias is not None else None

    def forward(self, x):
        base_output = self.linear(x)
        lora_output = self.lora(x)
        return base_output + lora_output


def apply_lora(model, target_layers=("attn1", "attn2", "proj"), rank=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and any(layer_name in name for layer_name in target_layers):
            model._modules[name] = LoRALinear(module, rank=rank)
        else:
            apply_lora(module, target_layers, rank)


class LoRATestModel(pl.LightningModule):
    def __init__(self, config_path: str = './models/cldm_v21.yaml', lora_rank: int = 8):
        super().__init__()
        self.save_hyperparameters()
        self.density_model = create_model(config_path)
        self.render_model = create_model(config_path)
        apply_lora(self.render_model, rank=lora_rank)
        self.render_model.save_dir = '/home/zitong/img2pano_cpy/'

    def apply_lora(self):
        apply_lora(self.render_model, rank=self.hparams.lora_rank)

    def test_step(self, batch, batch_idx):
        self.density_model.eval()
        self.render_model.eval()
        with torch.no_grad():
            self.density_model.shared_step(batch, model_name='density')
            loss, loss_dict = self.render_model.shared_step(batch, model_name='render')
            logger.log_img(self.render_model, batch, batch_idx, 'val')

        return {'psnr': PSNR, 'ssim': SSIM}


if __name__ == "__main__":
    model = LoRATestModel.load_from_checkpoint(
        checkpoint_path='./lora_fintung/last.ckpt',
        strict=False,
        lora_rank=8
    )

    test_dataset = MyDataset()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    logger = ImageLogger(batch_frequency=1)
    test_trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        callbacks=[logger]
    )

    test_trainer.test(model, dataloaders=test_loader)
