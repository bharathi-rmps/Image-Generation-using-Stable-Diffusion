import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk
from auth import token
import torch
from torch import autocast
from diffusers import StableDiffusion3Pipeline

app = tk.Tk()
app.geometry("600x500")
app.title("Image varuma?")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(app, width=580, height=40, font=("Arial", 16), text_color="Black", fg_color="White")
prompt.place(x=10, y=20)

lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

pipeline = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
        torch_dtype=torch.float32,
        text_encoder_3=None,
        tokenizer_3=None
    )
pipeline.enable_model_cpu_offload()

def image_generator(prompt):
    #pipeline.to(device)

    image = pipeline(
        prompt=prompt,
        negative_prompt="blurred, ugly, watermark, low, resolution, blurry",
        num_inference_steps=40,
        height=512,
        width=512,
        guidance_scale=9.0
    ).images[0]

    return image


def genImage():
    with autocast("cuda"):
        image = image_generator(prompt.get())
        
    image.save("vanthaImage.png")
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)

trigger = ctk.CTkButton(app, height=40, width=120, font=("Arial", 16), text_color="white", fg_color="blue", text="Kudra Image ah", command=genImage)
trigger.place(x=245, y=80)

app.mainloop()