import os
# Update the place where models are dumped for storage space control
os.environ['HF_HOME'] = os.path.join(os.getcwd(), "hf_home")

from diffusers import DiffusionPipeline


pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
image  = pipeline("An image of a squirrel in Picasso style").images[0]
image.save("image_of_squirrel_painting.png")