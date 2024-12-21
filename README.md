# I2I_diffusion_comparison

In order to run experiments you need to clone two repositories:

[EDM](https://github.com/NVlabs/edm)

[Guided-diffusion](https://github.com/openai/guided-diffusion)

Pretrained model for wild animals generation (wild64x64.pkl) can be found [here](https://drive.google.com/drive/folders/17s7C20TNbVo15BJA0hRwxH2112c1Fkcs)

Pretrained ImageNet classifier was taken from [here](https://github.com/openai/guided-diffusion).

To run experiments run notebooks in the following order:

1. setup_datasets.ipynb
2. run_methods.ipynb
3. visualize_results.ipynb
