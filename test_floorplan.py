import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from test_gradio_dataset import MyDataset
from cldm.model import create_model
from cldm.logger import ImageLogger


def mark_and_save_points(image_path, save_path):
    image = plt.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Click to mark points (close window to finish)")
    coords = []
    def onclick(event):
        if event.xdata and event.ydata:
            coords.append((int(event.xdata), int(event.ydata)))
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    plt.close()
    coords = np.array(coords)
    np.save(save_path, coords)

class CombinedModel(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-5):
        super().__init__()
        self.density_model = create_model('./models/cldm_v21.yaml')
        self.render_model = create_model('./models/cldm_v21.yaml')
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.render_model.save_dir = '/data/ZItong/img2pano_cpy/'

    def test_step(self, batch, batch_idx):
        self.density_model.eval()
        self.render_model.eval()
        with torch.no_grad():
            self.density_model.shared_step(batch, model_name='density')
            loss, loss_dict = self.render_model.shared_step(batch, model_name='render')

            logger.log_img(self.render_model, batch, batch_idx, 'val')


if __name__ == "__main__":
    image_path = './floor plan/color floor plan/2.jpg'
    save_path = './floor plan/marked_points.npy'
    if not os.path.exists(save_path):
        mark_and_save_points(image_path, save_path)
    model = CombinedModel.load_from_checkpoint(
        checkpoint_path='./state_dict_GCombinedDepthColorSeg/last.ckpt',
        strict=False
    )
    logger = ImageLogger(batch_frequency=1)
    pixel_scale = 85.333333333
    test_dataset = MyDataset(image_path, save_path, pixel_scale)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_trainer = pl.Trainer(accelerator='gpu', devices=[1], precision=32, callbacks=[logger])
    test_trainer.test(model, dataloaders=test_loader)

