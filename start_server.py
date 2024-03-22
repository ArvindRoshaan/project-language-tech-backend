from fastapi import FastAPI, File
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline 
import torch
from torch import autocast
from pyngrok import ngrok
import nest_asyncio
from fastapi.middleware.cors import CORSMiddleware
#from auth import auth_token
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
import uvicorn
from machine_translation import translate 
import time

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# loading the image generation model in advance
#modelid = "CompVis/stable-diffusion-v1-4"
#modelid = "nota-ai/bk-sdm-small"
modelid = "nota-ai/bk-sdm-tiny"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained(modelid).to(device)  #, use_auth_token=auth_token 

class TranslateRequest(BaseModel):
    text: str
    src: str
    tgt: str

# machine translation
@app.post("/translate_text/")
async def translate_text(text_request: TranslateRequest):
    print("Request payload: ", text_request.json())
    text = text_request.text
    src = text_request.src
    tgt = text_request.tgt
    if src == tgt:
        return {"translated_text": "Pick different languages!"}
    print(f"Received source sentence : {text}")
    print(f"Received source language : {src}")
    print(f"Received target sentence : {tgt}")
    translation = translate(text, src, tgt)
    with open("translated_text.txt", 'w') as f:
        f.write(f"{translation}")
    print("translation: ", translation)
    return {"translated_text": translation}

@app.get("/show_translated_text/")
async def get_translated_text():
    with open("translated_text.txt", 'r') as f:
        return f.read()


class ImageRequest(BaseModel):
    text: str
    lang: str

def generateImage(text, lang, image_path):
    if lang != "English":
        #need to translate text to English
        src = lang
        tgt = "English"
        text = translate(text, src, tgt)
        print("translated prompt: ", text)

    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with autocast(device): 
        #image = pipe(text, guidance_scale=8.5)["sample"][0]
        image = pipe(text, guidance_scale=8.5).images[0]
        image.save(image_path)

# image generation
image_path = "generated_image.png"
@app.post("/generate_image/")
async def generate_image(text_request: ImageRequest):
    print("Request payload: ", text_request.json())
    text = text_request.text
    lang = text_request.lang
    print("Received prompt: ", text)
    print(f"Received language: {lang}")
    generated_image = generateImage(text, lang, image_path)
    return FileResponse(image_path, media_type="image/png")
    # return {"received_text": text, "generated_image": generated_image}

@app.get("/show_generated_image/")
async def get_generated_image():
    return FileResponse(image_path, media_type="image/png")


ngrok_tunnel = ngrok.connect(8000)
print('Public URL: ', ngrok_tunnel.public_url)
# Run the FastAPI app
nest_asyncio.apply()
uvicorn.run(app, port=8000)