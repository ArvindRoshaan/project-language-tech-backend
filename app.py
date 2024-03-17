from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline 
import torch
from torch import autocast
from pyngrok import ngrok
import nest_asyncio
from fastapi.middleware.cors import CORSMiddleware
from auth import auth_token
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
import uvicorn

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 

class TextRequest(BaseModel):
    text: str

def generate(text): 
    with autocast(device): 
        image = pipe(text, guidance_scale=8.5)["sample"][0]
        image.save("P:\\Flutter_Projects\\generated_images\\generatedimage_new.png")
        return image

@app.post("/process_text/")
async def process_text(text_request: TextRequest):
    text = text_request.text
    print("Received text:", text)
    generated_image = generate(text)
    return {"received_text": text, "generated_image": generated_image}

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
# Run the FastAPI app
nest_asyncio.apply()
uvicorn.run(app, port=8000,reload=True)