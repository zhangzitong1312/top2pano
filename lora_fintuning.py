from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from lora_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CombinedModel(pl.LightningModule):
    def __init__(self,
                 config_path: str = './models/cldm_v21.yaml',
                 lora_rank: int = 8,
                 learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.density_model = create_model(config_path)
        self.render_model = create_model(config_path)
        apply_lora(self.render_model, rank=lora_rank)
        self.render_model.save_dir = '/home/zitong/top2pano/'
        self.learning_rate = learning_rate

    def _freeze_model(self, model, freeze_lora=True):
        for param in model.parameters():
            param.requires_grad = False  # Freeze everything

        if not freeze_lora:
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    for param in module.parameters():
                        param.requires_grad = True  # Only train LoRA

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.render_model.shared_step(batch, model_name='render')

        self.log_dict(loss_dict, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        lora_params = [p for name, p in self.render_model.named_parameters() if "lora" in name and p.requires_grad]
        return torch.optim.AdamW(lora_params, lr=self.learning_rate)



    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, **kwargs):
        model = super().load_from_checkpoint(
            checkpoint_path,
            strict=False,
            **kwargs
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model


# ================== 主程序 ==================
if __name__ == "__main__":
    model = CombinedModel.load_from_checkpoint(
        checkpoint_path='./state_dict_GCombinedDepthColorSeg/last.ckpt',
        lora_rank=8,
        learning_rate=1e-4
    )

    dataset = MyDataset("./lora_fintung")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./lora_fintung",
        filename="lock_last-checkpoint",
        save_last=True,
        save_top_k=1,
        monitor=None
    )

    logger = ImageLogger(batch_frequency=500)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[1],
        max_epochs=100,
        callbacks=[logger, checkpoint_callback]
    )

    trainer.fit(model, dataloader)