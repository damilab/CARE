import os
import argparse
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time
from datetime import timedelta
import random
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler
from cleanfid import fid
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel


def set_determinism(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def load_prompts_and_seeds(csv_path, prompt_col, seed_col, case_col,
                           limit=None, base_seed=42):
    df = pd.read_csv(csv_path)
    
    if prompt_col not in df.columns:
        raise ValueError(f"Column '{prompt_col}' not found in {csv_path}")
    prompts = df[prompt_col].astype(str).tolist()

    if seed_col in df.columns:
        raw = df[seed_col].tolist()
        seeds = []
        for i, s in enumerate(raw):
            try:
                seeds.append(int(s))
            except Exception:
                seeds.append(base_seed + i)
    else:
        seeds = [base_seed + i for i in range(len(prompts))]

    if case_col and case_col in df.columns:
        case_numbers = df[case_col].astype(str).tolist()
    else:
        case_numbers = [f"{i:06d}" for i in range(len(prompts))]

    if limit is not None and limit > 0:
        prompts = prompts[:limit]
        seeds = seeds[:limit]
        case_numbers = case_numbers[:limit]

    return prompts, seeds, case_numbers


def build_pipe(model_path, device, dtype=torch.float16):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, safety_checker=None, torch_dtype=dtype
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    return pipe


def maybe_load_unet(pipe, ckpt_path):
    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"UNet checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        pipe.unet.load_state_dict(state)
        pipe.unet.eval()


def generate_images_and_clip_batched(
    pipe, clip_model, clip_processor,
    prompts, seeds, case_numbers, out_dir,
    steps, guidance, height, width, device,
    gen_batch=25, clip_batch=25, desc="Generating"
):

    os.makedirs(out_dir, exist_ok=True)
    clip_scores = []
    clip_imgs_buf, clip_txts_buf = [], []

    N = len(prompts)
    start_time = time.time()
    pbar = tqdm(total=N, desc=desc, ncols=110)

    for start in range(0, N, gen_batch):
        end = min(start + gen_batch, N)
        batch_prompts = prompts[start:end]
        batch_seeds = seeds[start:end]
        batch_cases = case_numbers[start:end]
        batch_gens = [torch.Generator(device).manual_seed(int(s)) for s in batch_seeds]


        result = pipe(
            prompt=batch_prompts,
            generator=batch_gens,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
        )
        images = result.images 

        for k, img in enumerate(images):
            case_num = str(batch_cases[k])
            img = img.convert("RGB")
            img.save(os.path.join(out_dir, f"{case_num}.png"), compress_level=4, optimize=False)
            clip_imgs_buf.append(img)
            clip_txts_buf.append(batch_prompts[k])

            if len(clip_imgs_buf) == clip_batch:
                _scores = _clip_scores(clip_model, clip_processor, clip_txts_buf, clip_imgs_buf, device)
                clip_scores.extend(_scores)
                clip_imgs_buf.clear()
                clip_txts_buf.clear()

        pbar.update(end - start)

        elapsed = time.time() - start_time
        rate = (start + gen_batch) / elapsed if elapsed > 0 else 0
        eta = (N - (start + gen_batch)) / rate if rate > 0 else 0
        pbar.set_postfix({
            "elapsed": str(timedelta(seconds=int(elapsed))),
            "eta": str(timedelta(seconds=int(eta)))
        })

        torch.cuda.empty_cache()

    if clip_imgs_buf:
        _scores = _clip_scores(clip_model, clip_processor, clip_txts_buf, clip_imgs_buf, device)
        clip_scores.extend(_scores)
        clip_imgs_buf.clear()
        clip_txts_buf.clear()

    pbar.close()
    return sum(clip_scores) / max(len(clip_scores), 1)


def _clip_scores(clip_model, clip_processor, texts, images, device):
    inputs = clip_processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        img_feat = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_feat = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(p=2, dim=-1, keepdim=True)
    scores = (img_feat @ txt_feat.T).diag().detach().cpu().tolist()
    return scores


def compute_fid_ab(dirA, dirB, mode="clean", batch_size=64, num_workers=4, device="cuda"):
    return fid.compute_fid(
        dirA, dirB,
        mode=mode,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=True
    )
    
# For AdvUnlearn Model
# def load_pipeline_with_advunlearn(
#     device: str,
#     model_path: str = "CompVis/stable-diffusion-v1-4",
#     te_repo: str = "OPTML-Group/AdvUnlearn",
#     te_subfolder: str = "nudity_unlearned",
#     pipe_dtype: torch.dtype = torch.float16,
# ) -> StableDiffusionPipeline:
#     text_encoder = CLIPTextModel.from_pretrained(
#         te_repo,
#         subfolder=te_subfolder,
#         torch_dtype=torch.float16
#     ).to(device)

#     pipe = StableDiffusionPipeline.from_pretrained(
#         model_path,
#         safety_checker=None,
#         text_encoder=text_encoder,
#         torch_dtype=pipe_dtype
#     ).to(device)

#     te_dtype = pipe.unet.dtype
#     pipe.text_encoder = pipe.text_encoder.to(
#         device=pipe.device,
#         dtype=te_dtype
#     )

#     return pipe


def main(args):
    set_determinism(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    prompts, seeds, case_numbers = load_prompts_and_seeds(
        args.prompt_csv,
        args.prompt_column,
        args.seed_column,
        args.case_column,
        args.limit,
        base_seed=args.base_seed
    )

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # A: Original Stable Diffusion UNet
    out_A = os.path.join(args.output_dir, "SD")
    pipe_A = build_pipe(args.model_path, device)
    avg_clip_A = generate_images_and_clip_batched(
        pipe_A, clip_model, clip_processor,
        prompts, seeds, case_numbers, out_A,
        args.num_inference_steps,
        args.guidance_scale,
        args.height, args.width, device,
        gen_batch=args.gen_batch,
        clip_batch=args.clip_batch,
        desc="A: Generation (batch) + CLIP (batch)"
    )

    # B: UNet with checkpoint
    out_B = os.path.join(args.output_dir, "RECARE")
    pipe_B = build_pipe(args.model_path, device)
    maybe_load_unet(pipe_B, args.unet_checkpoint)
    avg_clip_B = generate_images_and_clip_batched(
        pipe_B, clip_model, clip_processor,
        prompts, seeds, case_numbers, out_B,
        args.num_inference_steps,
        args.guidance_scale,
        args.height, args.width, device,
        gen_batch=args.gen_batch,
        clip_batch=args.clip_batch,
        desc="B: Generation (batch) + CLIP (batch)"
    )
    
    out_COCO = args.coco_dir

    # FID computation
    print("\n[ FID ] Computing COCO vs original SD ...")
    fid_coco_A = compute_fid_ab(
        out_COCO, out_A,
        mode=args.fid_mode,
        batch_size=args.fid_batch,
        num_workers=args.num_workers,
        device="cpu"
    )

    print("\n[ FID ] Computing COCO vs checkpoint model ...")
    fid_coco_B = compute_fid_ab(
        out_COCO, out_B,
        mode=args.fid_mode,
        batch_size=args.fid_batch,
        num_workers=args.num_workers,
        device="cpu"
    )
    print(f"\nFID (COCO vs Original SD) [{args.fid_mode}]: {fid_coco_A:.5f}")
    print(f"\nFID (COCO vs Checkpoint Model) [{args.fid_mode}]: {fid_coco_B:.5f}")
    print(f"Average CLIP Score (Original SD): {avg_clip_A:.5f}")
    print(f"Average CLIP Score (Checkpoint Model): {avg_clip_B:.5f}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(f"FID (COCO vs Original SD) [{args.fid_mode}]: {fid_coco_A:.5f}\n")
        f.write(f"FID (COCO vs Checkpoint Model) [{args.fid_mode}]: {fid_coco_B:.5f}\n")
        f.write(f"Average CLIP Score (Original SD): {avg_clip_A:.5f}\n")
        f.write(f"Average CLIP Score (Checkpoint Model): {avg_clip_B:.5f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate two image sets under identical conditions, "
            "save images, compute FID, and report average CLIP score."
        )
    )
    parser.add_argument("--output_dir", type=str, default="../fid_results")
    parser.add_argument("--prompt_csv", type=str, default="../data/coco_30k/coco_30k.csv", help="Path to the reference COCO prompt CSV file")
    parser.add_argument("--coco_dir",type=str, default="../data/coco_30k/images/", help="Path to the reference COCO image directory")
    
    parser.add_argument("--prompt_column", type=str, default="prompt")
    parser.add_argument("--seed_column", type=str, default="evaluation_seed")
    parser.add_argument("--case_column", type=str, default="case_number")
    parser.add_argument("--limit", type=int, default=30000)
    parser.add_argument("--base_seed", type=int, default=42)

    parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--unet_checkpoint", type=str,
                        default="/recare_weights/nudity/ReCARE-Diffusers-UNet.pt")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)

    parser.add_argument("--gen_batch",type=int,default=25,help="Batch size for image generation (prompts/seeds processed together)")
    parser.add_argument("--clip_batch",type=int,default=25,help="Batch size for CLIP evaluation")

    parser.add_argument("--fid_mode",type=str,default="clean",choices=["clean", "legacy_pytorch"])
    parser.add_argument("--fid_batch", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    main(args)