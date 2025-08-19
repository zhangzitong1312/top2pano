from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Import your dataset class here
#from top_down_dataset_gibson import MyDataset
from top_down_dataset_matterport import MyDataset

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning.callbacks import ModelCheckpoint
import torch


# Configs
resume_path = './models/control_sd21_ini.ckpt'
batch_size = 3
logger_freq = 1
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

checkpoint_callback = ModelCheckpoint(
    dirpath="./state_dict",
    filename="lock_last-checkpoint",
    save_last=True,
    save_top_k=1,
    monitor=None
)


class CombinedModel(pl.LightningModule):
    def __init__(self, density_model, render_model, learning_rate):
        super(CombinedModel, self).__init__()
        self.density_model = density_model
        self.render_model = render_model
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        self.density_model.shared_step(batch, model_name='density')
        loss, loss_dict = self.render_model.shared_step(batch,model_name='render')

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.density_model.parameters()) + list(self.render_model.parameters()),
            lr=self.learning_rate
        )
        return optimizer

density_model = create_model('./models/cldm_v21.yaml')
density_model.load_state_dict(load_state_dict(resume_path, location='cpu'),strict=False)
density_model.learning_rate = learning_rate
density_model.sd_locked = sd_locked
density_model.only_mid_control = only_mid_control

render_model = create_model('./models/cldm_v21.yaml')
render_model.load_state_dict(load_state_dict(resume_path, location='cpu'),strict=False)
render_model.learning_rate = learning_rate
render_model.sd_locked = sd_locked
render_model.only_mid_control = only_mid_control


dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
combined_model = CombinedModel(density_model, render_model, learning_rate)
trainer = pl.Trainer(accelerator='gpu', devices=[0], precision=32, callbacks=[logger, checkpoint_callback], max_epochs=1)
trainer.fit(combined_model, dataloader)
