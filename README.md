# Top2Pano: Learning to Generate Indoor Panoramas from Top-Down View

![teaser](assets/teaser.png)

# Environment setup and downloading the pretrained model

## Step 1. Create a conda environment

    conda env create -f environment.yaml
    conda activate top2pano

## Step 2. Downloading the pretrained model
Please follow the <a href="https://github.com/lllyasviel/ControlNet" target="_blank" rel="noopener noreferrer">ControlNet</a> instructions, download the pretrained model, and place it in the <code>models</code> folder.

# Training model

    python main.py

You can replace the dataset file to run the Gibson or Matterport dataset.

# Inference with checkpoint

    python test.py

You can replace the dataset file to test on the Gibson or Matterport dataset.


# Generate panoramas on the floor plan

    python test_floorplan.py


# Prepare the Dataset


# Acknowledgements

This project builds upon several excellent open source projects:

* [ControlNet](https://github.com/lllyasviel/ControlNet) - A neural network structure that adds spatial conditioning to diffusion models, enabling precise control over image generation with external guidance.  


* [Sat2Density](https://github.com/qianmingduowan/Sat2Density) - A geometric-based end-to-end framework for synthesizing ground-view panoramas from satellite imagery.

We thank the authors and contributors of these projects for their valuable contributions to the open-source community.


# Citation
```
@inproceedings{zhang2025top2pano,
  title     = {Top2Pano: Learning to Generate Indoor Panoramas from Top-Down View},
  author    = {Zitong Zhang and Suranjan Gautam and Rui Yu},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```










