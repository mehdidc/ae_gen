Release  of ConvAE model (from [here](https://arxiv.org/pdf/1606.04345.pdf)) and DeepConvAE model (from [here](https://tel.archives-ouvertes.fr/tel-01838272/file/75406_CHERTI_2018_diffusion.pdf), Section 10.1 with `L=3`)

# Install requirements

`pip install -r requirements.txt`

# Download models

```bash
bash download_models.sh
```

# Training

`python cli.py train  --dataset=mnist --folder=mnist --model=convae`

`python cli.py train  --dataset=mnist --folder=mnist --model=deep_convae`

# Generate samples

```bash
python cli.py test --model-path=convae.th --nb-generate=100 --folder=convae
```

```bash
python cli.py test --model-path=deep_convae.th --nb-generate=100 --folder=deep_convae
```
