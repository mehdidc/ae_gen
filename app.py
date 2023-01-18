import torch
import torchvision
import gradio as gr
from PIL import Image
from cli import iterative_refinement
from viz import grid_of_images_default
from subprocess
subprocess.call("download_models.sh", shell=True)
models = {
    "convae": torch.load("convae.th", map_location="cpu"),
    "deep_convae": torch.load("deep_convae.th", map_location="cpu"),
}

def gen(model, seed, nb_iter, nb_samples, width, height):
    torch.manual_seed(int(seed))
    bs = 64
    model = models[model]
    samples = iterative_refinement(
        model, 
        nb_iter=int(nb_iter),
        nb_examples=int(nb_samples), 
        w=int(width), h=int(height), c=1, 
        batch_size=bs,
    )
    grid = grid_of_images_default(samples.reshape((samples.shape[0]*samples.shape[1], int(height), int(width), 1)).numpy(), shape=(samples.shape[0], samples.shape[1])) 
    grid = (grid*255).astype("uint8")
    return Image.fromarray(grid)

iface = gr.Interface(
    fn=gen,
    inputs=[gr.Dropdown(list(models.keys()), value="deep_convae"), gr.Number(value=0), gr.Number(value=20), gr.Number(value=1), gr.Number(value=28), gr.Number(value=28)],
    outputs="image"
)
iface.launch()
