import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from top_down_dataset_gibson_test import MyDataset
#from top_down_dataset_matterport_test import MyDataset

from cldm.model import create_model
from cldm.logger import ImageLogger


class CombinedModel(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-5):
        super().__init__()
        self.density_model = create_model('./models/cldm_v21.yaml')
        self.render_model = create_model('./models/cldm_v21.yaml')
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.render_model.save_dir = '/home/zitong/top2pano/'



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.density_model.parameters()) + list(self.render_model.parameters()),
            lr=self.learning_rate
        )
        return optimizer

    def test_step(self, batch, batch_idx):
        self.density_model.eval()
        self.render_model.eval()

        with torch.no_grad():
            self.density_model.shared_step(batch, model_name='density')
            loss, loss_dict = self.render_model.shared_step(batch, model_name='render')
            logger.log_img(self.render_model, batch, batch_idx, 'val')



if __name__ == "__main__":
    model = CombinedModel.load_from_checkpoint(
        checkpoint_path='./state_dict_MCombinedDepthColorSeg/last.ckpt',
        strict=False
    )

    logger = ImageLogger(batch_frequency=1)
    test_dataset = MyDataset()
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_trainer = pl.Trainer(accelerator='gpu', devices=[1], precision=32)
    test_trainer.test(model, dataloaders=test_loader)


