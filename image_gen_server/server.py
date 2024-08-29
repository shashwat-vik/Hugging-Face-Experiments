from flask import Flask, render_template, request
from uuid import uuid4


#################################################
import os
# Update the place where models are dumped for storage space control
os.environ['HF_HOME'] = os.path.join(os.getcwd(), "../hf_home")

from diffusers import AmusedPipeline
import torch

pipeline = AmusedPipeline.from_pretrained("amused/amused-256", variant="fp16", torch_dtype=torch.float16)
pipeline.vqvae.to(torch.float32)  # vqvae is producing nans n fp16
pipeline = pipeline.to("cuda")

def textToImage(prompt):
    image = pipeline(prompt, generator=torch.Generator('cuda').manual_seed(8)).images[0]
    return image
#################################################


app = Flask(__name__)


@app.route("/")
def root():
    return render_template("server.html")


@app.route("/gen_image", methods=['POST'])
def api():
    prompt = request.json['prompt']
    pil_img = textToImage(prompt)
    file_path = "static/temp/%s.png" % uuid4().hex[:4]
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    pil_img.save(file_path, 'PNG')
    return file_path


if __name__ == '__main__':
    app.run(port=8000, debug=True)