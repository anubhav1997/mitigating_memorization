from diffusers import DiffusionPipeline
import torch
from diffusers import DDPMScheduler, UNet2DModel, FlaxKarrasVeScheduler
from PIL import Image
import torch
import numpy as np 
from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDPMScheduler, HeunDiscreteScheduler, KarrasVeScheduler, EDMEulerScheduler, DPMSolverMultistepScheduler #FlaxKarrasVeOutput
from tqdm.auto import tqdm
from npy_append_array import NpyAppendArray
import os 
from transformers import AutoTokenizer
import json 
from scipy.signal import argrelextrema

import time 


pretrained_path = "CompVis/stable-diffusion-v1-4"

vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae"#, use_safetensors=True
                                   ) #stabilityai/stable-diffusion-2-1
tokenizer = AutoTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer", use_fast=False) #CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_path, subfolder="text_encoder"#, use_safetensors=True
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_path, subfolder="unet"#, use_safetensors=True
)
scheduler = DPMSolverMultistepScheduler.from_pretrained(pretrained_path, subfolder="scheduler")  #KarrasVeScheduler.from_pretrained("CompVis/stable-diffusion-v1-4")
# scheduler.set_timesteps(50)
# scheduler.use_karras_sigmas = True

torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

height = 512 #256 # default height of Stable Diffusion
width = 512 #256  # default width of Stable Diffusion
num_inference_steps = 50  # Number of denoising steps
guidance_scale = 2 #7.5 # Scale for classifier-free guidance
generator = torch.manual_seed(42)#.to(torch_device)  # Seed generator to create the initial latent noise
torch.cuda.manual_seed_all(42)
n_samples = 500 #10000
batch_size = 1
use_json = True 
CFG_scheduling = 'static'
cads = False #True 

prompt_augmentation_ = None #"rand_word_add"
outdir = "SDv1_500_mem_webster_dynamic_transition_cfg1" 

if not os.path.exists(outdir):
    os.makedirs(outdir)

if not os.path.exists(outdir + "/Images/"):
    os.makedirs(outdir + "/Images/")


if use_json == True:

    with open('sdv1_500_memorized.jsonl', 'r') as json_file:
        json_list = list(json_file)
    text_laion = []
    
    for json_str in json_list:
        result = json.loads(json_str)        
        text_laion.append(result)



def find_min_max_points(latents_init, text_embeddings):
    
    scheduler.set_timesteps(num_inference_steps)
    
    latents = latents_init * scheduler.init_noise_sigma

    diffs = []
    
    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
    
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        
        text_embeddings_final = text_embeddings
        
        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_final).sample
    
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        diff_current = torch.norm(noise_pred_uncond-noise_pred_text)
        diffs.append(diff_current.item())

        noise_pred = noise_pred_uncond #+ guidance_scale_new * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t,latents).prev_sample

    min_indexes = argrelextrema(np.array(diffs), np.less)   
    max_indexes = argrelextrema(np.array(diffs), np.greater)

    print(min_indexes, max_indexes)
    
    return min_indexes[0], max_indexes[0], diffs



def insert_rand_word(sentence,word):
    import random
    sent_list = sentence.split(' ')
    sent_list.insert(random.randint(0, len(sent_list)), word)
    new_sent = ' '.join(sent_list)
    return new_sent

def prompt_augmentation(prompt, aug_style,tokenizer=None, repeat_num=2):
    
    if aug_style =='rand_numb_add':
        for i in range(repeat_num):
            randnum  = np.random.choice(100000)
            prompt = insert_rand_word(prompt,str(randnum))
    elif aug_style =='rand_word_add':
        for i in range(repeat_num):
            randword = tokenizer.decode(list(np.random.randint(49400, size=1)))
            prompt = insert_rand_word(prompt,randword)
    elif aug_style =='rand_word_repeat':
        wordlist = prompt.split(" ")
        for i in range(repeat_num):
            randword = np.random.choice(wordlist)
            prompt = insert_rand_word(prompt,randword)
    else:
        raise Exception('This style of prompt augmnentation is not written')
    return prompt
    



start_time = time.time()

i = 0

while i < n_samples and i < len(text_laion):

    if i+ batch_size > n_samples:
        batch_size = n_samples - i

    if use_json == False:
        prompt = ["" for i in range(batch_size)]
    else:
        prompt = text_laion[i:i+batch_size]
        prompt = [p['caption'] for p in prompt]

    

    if prompt is None or len(prompt) == 0:
        prompt = " "

    if prompt_augmentation_ is not None:
        prompt = [prompt_augmentation(prompt[0], prompt_augmentation_, tokenizer=tokenizer,repeat_num=4)]

    
    
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents_init = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        # generator=generator,
        device=torch_device,
    )

    # min_indexes, max_indexes =  find_min_max_points(latents_init, text_embeddings)

    scheduler.set_timesteps(num_inference_steps)
    
    # for k in scheduler.timesteps:
        
    latents = latents_init * scheduler.init_noise_sigma
    
    # scheduler.set_timesteps(num_inference_steps)

    transition_point = -1 
    diff_value_prev = -1 
    diff_value_prev_prev = -1 
    prev_latent = latents
    
    diffs = []
    j =0
    ts = scheduler.timesteps
    # for t in tqdm(scheduler.timesteps):
    
    while j < len(ts):
        t = ts[j]
        
    
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
    
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        if cads == True:
            s = 0.1 
            phi = 1
            tau1 = 500 #600
            tau2 = 800
            
            gamma = 0
            if t<=tau1:
                gamma = 1
            elif t> tau1 and t<= tau2:
                gamma = (tau2 - t)/float(tau2 - tau1)
                
            text_embeddings_t = (gamma**0.5)*text_embeddings + s*((1-gamma)**0.5) * torch.rand(text_embeddings.shape, device=text_embeddings.device)
            text_embeddings_t_rescaled = (text_embeddings_t - torch.mean(text_embeddings_t))* torch.std(text_embeddings)/ torch.std(text_embeddings_t) + torch.mean(text_embeddings)
            text_embeddings_final = phi * text_embeddings_t_rescaled + (1-phi)*text_embeddings_t
            
        else:
            text_embeddings_final = text_embeddings
        
        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_final).sample
    
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        diff_current = torch.norm(noise_pred_uncond-noise_pred_text) #1- torch.nn.functional.cosine_similarity(torch.flatten(noise_pred_text), torch.flatten(noise_pred_uncond), dim=0) #
        diffs.append(diff_current.item())

        min_index = argrelextrema(np.array(diffs), np.less)
        # print(len(min_index[0]), min_index, scheduler.timesteps[min_index], t)
        
        # print(transition_point, diff_current, diff_value_prev, diff_value_prev_prev, t)
        
        if transition_point ==-1 and  diff_current>diff_value_prev and  diff_value_prev_prev>diff_value_prev: #len(min_index[0])!=0: # #(diff_value_prev - diff_current) < (diff_value_prev_prev - diff_value_prev):
            
            transition_point = t
            guidance_scale = 7.5
            CFG_scheduling = 'static'
            
        elif transition_point != -1:
            guidance_scale = 7.5
            CFG_scheduling = 'static'
        else:
            guidance_scale = 1.0
            CFG_scheduling = 'static'
        
        
        diff_value_prev_prev = diff_value_prev
        diff_value_prev = diff_current
        
        
        if CFG_scheduling == 'invlinear':
            guidance_scale_new = guidance_scale * (t/1000)
        elif CFG_scheduling == 'linear':
            guidance_scale_new = guidance_scale * (1 - (t/1000))
        elif CFG_scheduling == 'cosine':
            pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
            guidance_scale_new = guidance_scale * (torch.cos(pi*t/1000).item() + 1)
        elif CFG_scheduling == 'sine':
            pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
            guidance_scale_new = guidance_scale * (torch.sin(pi*(t)/1000 - pi/2).item() + 1)
        elif CFG_scheduling == 'v_shape':
            
            if t>500:
                guidance_scale_new = guidance_scale * (1 - (t/1000))
            else:
                guidance_scale_new = guidance_scale * (t/1000)
        elif CFG_scheduling == 'a_shape':
            
            if t<500:
                guidance_scale_new = guidance_scale * (1 - (t/1000))
            else:
                guidance_scale_new = guidance_scale * (t/1000)
                
        elif CFG_scheduling == 'static':
            guidance_scale_new = guidance_scale
        
        noise_pred = noise_pred_uncond + guidance_scale_new * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        prev_latents = latents
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        j+=1
        
    # scale and decode the image latents with vae  
    latents = 1./ vae.config.scaling_factor * latents
    
    
    with torch.no_grad():
        images = vae.decode(latents).sample

    for j in range(batch_size):
        image = images[j]
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        
        image.save(f"{outdir}/Images/{i+j}_{transition_point}.png")
    
    i += batch_size 
        
print("--- %s seconds ---" % (time.time() - start_time))

