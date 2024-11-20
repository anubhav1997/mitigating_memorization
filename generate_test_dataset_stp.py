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
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--use_json",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/scratch/aj3281/DCR/DCR/sd-finetuned-org_parameters_instancelevel_blip_nodup_laion/checkpoint/",
    )
    parser.add_argument(
        "--start_iter",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end_iter",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="LAION_10k_DCR_with_text_100000_0_500_7_5_else_cosine_1",
    )
    
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
    )
    parser.add_argument(
        "--guidance_change_step",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--guidance_scale_later",
        type=float,
        default=7.5,
    )
    parser.add_argument(
        "--CFG_scheduling",
        type=str,
        default="static",
    )
    parser.add_argument(
        "--CFG_scheduling_later",
        type=str,
        default="static",
    )
    
    parser.add_argument(
        "--cads",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="DPM",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--prompt_json",
        type=str,
        default='/scratch/aj3281/DCR/DCR/data/laion_10k_random/laion_combined_captions.json',
    )
    parser.add_argument(
        "--prompt_augmentation",
        type=str,
        default=None,
    )
    return parser.parse_args()
    
args = parse_args()



pretrained_path = args.pretrained_model_name_or_path


vae = AutoencoderKL.from_pretrained(pretrained_path, subfolder="vae"#, use_safetensors=True
                                   ) #stabilityai/stable-diffusion-2-1
tokenizer = AutoTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer", use_fast=False) #CLIPTokenizer.from_pretrained(pretrained_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_path, subfolder="text_encoder"#, use_safetensors=True
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_path, subfolder="unet"#, use_safetensors=True
)
scheduler = DPMSolverMultistepScheduler.from_pretrained(pretrained_path, subfolder="scheduler")  

torch_device = "cuda"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)

height = args.img_size
width = args.img_size
num_inference_steps = 50  # Number of denoising steps
guidance_scale = args.guidance_scale # Scale for classifier-free guidance
generator = torch.manual_seed(42)  # Seed generator to create the initial latent noise
torch.cuda.manual_seed_all(42)


n_samples = args.end_iter
batch_size = 1
use_json = args.use_json
CFG_scheduling = args.CFG_scheduling
cads = args.cads
outdir = args.outdir

if not os.path.exists(outdir):
    os.makedirs(outdir)

if not os.path.exists(outdir + "/Images/"):
    os.makedirs(outdir + "/Images/")


if use_json == True:

    prompt_json = args.prompt_json
    with open(prompt_json) as f:
        all_prompts_dict = json.load(f)
    
    text_laion = [v[0] for k,v in all_prompts_dict.items()]
    np.random.seed(42)
    text_laion = list(np.random.choice(text_laion,n_samples))


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
    



i = args.start_iter

while i < n_samples:

    if i+ batch_size > n_samples:
        batch_size = n_samples - i

    if use_json == False:
        prompt = ["" for i in range(batch_size)]
    else:
        prompt = text_laion[i:i+batch_size]
        # prompt = [p['caption'] for p in prompt]

    scheduler.set_timesteps(num_inference_steps)

    if prompt is None or len(prompt) == 0:
        prompt = " "

    if args.prompt_augmentation is not None:
        prompt = [prompt_augmentation(prompt[0], args.prompt_augmentation, tokenizer=tokenizer,repeat_num=4)]

    
    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]
    
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        # generator=generator,
        device=torch_device,
    )
    
    latents = latents * scheduler.init_noise_sigma


    ts = scheduler.timesteps
    t_index = 0
    
    while t_index < len(ts):
        t = ts[t_index]
        
        t_index += 1
    #for t in tqdm(scheduler.timesteps):
        
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
    
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        
        text_embeddings_final = text_embeddings
        
        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_final).sample
    
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        if t == args.guidance_change_step:
            # print(scheduler._step_index, t_index-1)
            scheduler.set_timesteps(200)
            ts = scheduler.timesteps
            t_index = min(range(len(ts)), key=lambda i: abs(ts[i]-t))
            scheduler._step_index = t_index - 1 
            t = ts[t_index]

        if t>args.guidance_change_step:
            CFG_scheduling = args.CFG_scheduling
            guidance_scale = args.guidance_scale
            
        else:
            CFG_scheduling = args.CFG_scheduling_later
            guidance_scale = args.guidance_scale_later

        
        if CFG_scheduling == 'invlinear':
            guidance_scale_new = guidance_scale * (t/1000)
        elif CFG_scheduling == 'linear':
            guidance_scale_new = guidance_scale * (1 - (t/1000))
        elif CFG_scheduling == 'cosine':
            pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
            # print(pi)
            # exit(0)
            guidance_scale_new = guidance_scale * (torch.cos(pi*t/1000).item() + 1)
        elif CFG_scheduling == 'sine':
            pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
            guidance_scale_new = guidance_scale * (torch.sin(pi*t/1000 - pi/2).item() + 1)
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
        latents = scheduler.step(noise_pred, t,latents).prev_sample
    
    
    # scale and decode the image latents with vae
    
    latents = 1./ vae.config.scaling_factor * latents

    
    with torch.no_grad():
        images = vae.decode(latents).sample

    for j in range(batch_size):
        image = images[j]
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        
        image.save(f"{outdir}/Images/{i+j}.png")
    
    i += batch_size 
        
