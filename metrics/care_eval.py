import os
import csv
import argparse
import re
from typing import List

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel


# LABELS:
# - COCO-style base object categories
# - plus CARE-specific target concepts used for evaluation
#   (e.g., 'stars', 'freshwater' as CARE evaluation targets beyond COCO objects)
LABELS = [
    'bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table',
    'toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush','person','stars','freshwater'
]
NAME2IDX = {n: i for i, n in enumerate(LABELS)}


def set_seed(seed: int, device: str = "cuda") -> None:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def append_csv(path: str, row: dict, fieldnames: List[str]) -> None:
    if not path:
        raise ValueError("--csv_summary must be provided.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_clip(model_name: str, device: str):
    model = CLIPModel.from_pretrained(model_name).to(device)
    proc = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, proc


@torch.no_grad()
def build_text_features(clip_model, clip_proc, device: str, use_template: bool = False) -> torch.Tensor:
    def to_template(name: str) -> str:
        vowels = set("aeiou")
        no_article = {
            "scissors","skis","snowboard","sports ball","baseball bat","baseball glove",
            "skateboard","surfboard","tennis racket","wine glass","potted plant",
            "dining table","cell phone","teddy bear","hair drier","tv","keyboard",
            "microwave","refrigerator"
        }
        if name in no_article or (name.endswith("s") and name not in {"glass"}):
            return f"A photo of {name}"
        article = "an" if name[0].lower() in vowels else "a"
        return f"A photo of {article} {name}"

    labels_txt = [to_template(n) for n in LABELS] if use_template else LABELS
    toks = clip_proc(text=labels_txt, return_tensors="pt", padding=True).to(device)
    feats = clip_model.get_text_features(**toks)
    return feats / feats.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_scores_batch(
    clip_model,
    clip_proc,
    images: List[Image.Image],
    text_feats: torch.Tensor,
    target_idx: int,
    device: str
):
    inputs = clip_proc(images=images, return_tensors="pt").to(device)
    img_feats = clip_model.get_image_features(**inputs)
    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

    sims = img_feats @ text_feats.T
    pred_idxs = sims.argmax(dim=1)
    target_scores = sims[:, target_idx]
    return pred_idxs.tolist(), target_scores.tolist()


def load_pipeline(
    device: str,
    sd_model_path: str,
    unet_checkpoint: str | None = None,
    te_repo: str | None = None,
    te_subfolder: str | None = None,
    te_cache: str | None = None,
    dtype_when_cuda=torch.float16
):
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    pipe_dtype = dtype_when_cuda if use_cuda else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        sd_model_path,
        safety_checker=None,
        torch_dtype=pipe_dtype,
    ).to(device)

    if te_repo and te_subfolder:
        te = CLIPTextModel.from_pretrained(
            te_repo,
            subfolder=te_subfolder,
            cache_dir=te_cache,
            torch_dtype=pipe_dtype,
        ).to(device)
        pipe.text_encoder = te.to(device=device, dtype=pipe.unet.dtype)

    if unet_checkpoint:
        print(f"[UNet] Loading checkpoint: {unet_checkpoint}")
        sd = torch.load(unet_checkpoint, map_location="cpu")

        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "unet" in sd and isinstance(sd["unet"], dict):
            sd = sd["unet"]

        def strip_prefix(d: dict, p: str) -> dict:
            return {(k[len(p):] if k.startswith(p) else k): v for k, v in d.items()}

        sd = strip_prefix(sd, "module.")
        sd = strip_prefix(sd, "model.diffusion_model.")
        sd = strip_prefix(sd, "unet.")

        missing, unexpected = pipe.unet.load_state_dict(sd, strict=False)
        print(f"[UNet] missing={len(missing)}, unexpected={len(unexpected)}")
        if missing[:5]:
            print("[UNet] sample missing:", missing[:5])
        if unexpected[:5]:
            print("[UNet] sample unexpected:", unexpected[:5])

    return pipe


def slugify(s: str, maxlen: int = 60) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-\.]", "", s)
    return s[:maxlen] if s else "prompt"


def gather_prompts(gen_prompt: str, gen_prompts_file: str) -> List[str]:
    if gen_prompts_file:
        prompts = []
        with open(gen_prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
        if not prompts:
            raise ValueError("No prompts found in --gen_prompts_file.")
        return prompts

    if gen_prompt.strip():
        return [gen_prompt.strip()]

    raise ValueError("Provide --gen_prompt or --gen_prompts_file.")


def main():
    ap = argparse.ArgumentParser(description="CARE score evaluation via SD generation + CLIP classification.")

    # SD pipeline
    ap.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--unet_checkpoint", type=str, default="", help="Optional UNet .pt checkpoint path")

    # Optional text-encoder override (loader supports for AdvUnlearn)
    ap.add_argument("--te_repo", type=str, default="")
    ap.add_argument("--te_subfolder", type=str, default="")
    ap.add_argument("--te_cache", type=str, default="")

    # Prompt source
    ap.add_argument("--gen_prompt", type=str, default="", help="Single prompt")
    ap.add_argument("--gen_prompts_file", type=str, default="", help="Text file: one prompt per line")

    # Evaluation
    ap.add_argument("--targets", type=str, default="person",
                    help="Comma-separated labels (must be in LABELS)")
    ap.add_argument("--num_images", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_template_text", action="store_true", help="Use 'A photo of (a/an) ...' template for CLIP text")

    # Output
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--out_dir", type=str, default="care_results")
    ap.add_argument("--csv_summary", type=str, default="")

    # CLIP
    ap.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")

    args = ap.parse_args()

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    set_seed(args.seed, device)

    os.makedirs(args.out_dir, exist_ok=True)
    args.csv_summary = os.path.join(args.out_dir, "summary.csv")
    
    print(f"[Device] {device}")
    print(f"[SD] model_path={args.model_path}")
    if args.unet_checkpoint:
        print(f"[SD] unet_checkpoint={args.unet_checkpoint}")
    print(f"[Eval] targets={args.targets}, n={args.num_images}, bs={args.batch_size}, seed={args.seed}")

    pipe = load_pipeline(
        device=device,
        sd_model_path=args.model_path,
        unet_checkpoint=(args.unet_checkpoint or None),
        te_repo=(args.te_repo or None),
        te_subfolder=(args.te_subfolder or None),
        te_cache=(args.te_cache or None),
        dtype_when_cuda=torch.float16,
    )
    clip_model, clip_proc = load_clip(args.clip_model, device)
    text_feats = build_text_features(clip_model, clip_proc, device, use_template=args.use_template_text)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    for t in targets:
        if t not in NAME2IDX:
            raise ValueError(f"Unknown label: {t} (not in LABELS)")

    prompt_list = gather_prompts(args.gen_prompt, args.gen_prompts_file)
    print(f"[Prompts] {len(prompt_list)} prompts loaded")

    fieldnames = ["prompt", "target", "n", "correct_top1", "acc_top1", "mean_target_score"]

    total = args.num_images
    bs = max(1, args.batch_size)

    overall_n = 0
    overall_correct = 0
    overall_target_score_sum = 0.0

    for prompt_str in prompt_list:
        pslug = slugify(prompt_str)

        for tname in targets:
            target_idx = NAME2IDX[tname]
            correct = 0
            target_score_sum = 0.0

            for start in range(0, total, bs):
                cur_bs = min(bs, total - start)
                prompts = [prompt_str] * cur_bs

                gens = []
                for i in range(cur_bs):
                    g = torch.Generator(device=device)
                    g.manual_seed(args.seed + (start + i))
                    gens.append(g)

                out = pipe(
                    prompt=prompts,
                    generator=gens,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                )
                imgs = out.images

                pred_idxs, target_scores = clip_scores_batch(
                    clip_model, clip_proc, imgs, text_feats, target_idx, device
                )

                for i, (pi, ts) in enumerate(zip(pred_idxs, target_scores)):
                    correct += int(pi == target_idx)
                    target_score_sum += float(ts)

                    if args.save_images:
                        save_dir = os.path.join(args.out_dir, pslug)
                        os.makedirs(save_dir, exist_ok=True)
                        img_path = os.path.join(save_dir, f"{tname}_seed{args.seed}_{start+i:04d}.png")
                        imgs[i].save(img_path)

            acc = correct / total
            mean_target = target_score_sum / total

            append_csv(
                args.csv_summary,
                {
                    "prompt": prompt_str,
                    "target": tname,
                    "n": total,
                    "correct_top1": correct,
                    "acc_top1": round(acc, 6),
                    "mean_target_score": round(mean_target, 6),
                },
                fieldnames,
            )

            print(f"[prompt='{prompt_str}' | {tname}] acc@1={acc:.4f} ({correct}/{total}), "
                  f"mean_target_score={mean_target:.4f}")

            overall_n += total
            overall_correct += correct
            overall_target_score_sum += target_score_sum

    if overall_n > 0:
        overall_acc = overall_correct / overall_n
        overall_mean_target = overall_target_score_sum / overall_n

        append_csv(
            args.csv_summary,
            {
                "prompt": "__ALL__",
                "target": "__ALL__",
                "n": overall_n,
                "correct_top1": overall_correct,
                "acc_top1": round(overall_acc, 6),
                "mean_target_score": round(overall_mean_target, 6),
            },
            fieldnames,
        )

        print(f"[OVERALL] acc@1={overall_acc:.4f} ({overall_correct}/{overall_n}), "
              f"mean_target_score={overall_mean_target:.4f}")


if __name__ == "__main__":
    main()
