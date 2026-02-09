from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os, argparse
import torch
import pandas as pd
from tqdm import tqdm

def get_image_paths(folder):
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def main():
    parser = argparse.ArgumentParser(
        prog="ImageClassification",
        description="Classify images with ResNet50 and save top-k predictions."
    )
    parser.add_argument('--folder_path', help='path to images', type=str, required=True)
    parser.add_argument('--save_path', help='path to save results', type=str, required=False, default=None)
    parser.add_argument('--device', type=str, required=False, default='cuda:0')
    parser.add_argument('--topk', type=int, required=False, default=5)
    parser.add_argument('--batch_size', type=int, required=False, default=250)
    parser.add_argument('--gt_label', type=str, required=False, default="tench", 
                        help='ground truth label (e.g., "tench")')
    parser.add_argument('--gt_index', type=int, required=False, default=None, 
                        help='ground truth index (ImageNet class id, 0~999)')
    args = parser.parse_args()

    folder = args.folder_path
    save_path = args.save_path
    device = args.device
    topk = max(1, args.topk)
    batch_size = max(1, args.batch_size)

    if save_path is None:
        name_ = os.path.basename(os.path.normpath(folder))
        save_path = os.path.join(folder, f"{name_}_classification.csv")

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device).eval()
    preprocess = weights.transforms()

    image_paths = get_image_paths(folder)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found under: {folder}")

    tensors = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        tensors.append(preprocess(img))
    images = torch.stack(tensors) 

    cols = {"path": image_paths}
    for k in range(1, topk + 1):
        cols[f"index_top{k}"] = []
        cols[f"category_top{k}"] = []
        cols[f"scores_top{k}"] = []

    n = len(image_paths)
    num_batches = (n - 1) // batch_size + 1

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Processing Batches"):
            b = images[i * batch_size: min(n, (i + 1) * batch_size)].to(device)
            prob = model(b).softmax(1)                        
            probs, cls_ids = torch.topk(prob, topk, dim=1)      

            cls_ids = cls_ids.cpu().numpy()
            probs = probs.cpu().numpy()

            for k in range(1, topk + 1):
                ids_k = cls_ids[:, k - 1]
                probs_k = probs[:, k - 1]
                cols[f"index_top{k}"].extend(ids_k)
                cols[f"scores_top{k}"].extend(probs_k)
                cols[f"category_top{k}"].extend([weights.meta["categories"][int(x)] for x in ids_k])

    df = pd.DataFrame(cols)

    total = len(df)

    # Case 1: compare using ground-truth label string
    if args.gt_label is not None:
        gt_label = args.gt_label.lower()
        preds_top1 = df["category_top1"].str.lower()
        correct_top1 = (preds_top1 == gt_label).sum()
        asr_top1 = correct_top1 / total * 100
        print(f"[ASR] Top-1 (label): {asr_top1:.2f}% ({correct_top1}/{total})")

        # Top-5 accuracy
        correct_top5 = 0
        matched_paths = []
        for _, row in df.iterrows():
            preds = [row[f"category_top{k}"].lower() for k in range(1, 6)]
            if gt_label in preds:
                correct_top5 += 1
                matched_paths.append(row["path"])
        asr_top5 = correct_top5 / total * 100
        print(f"[ASR] Top-5 (label): {asr_top5:.2f}% ({correct_top5}/{total})")

        if matched_paths:
            print("Matched Top-5 images:")
            for p in matched_paths:
                print("  ", p)

    # Case 2: compare using ground-truth class index
    if args.gt_index is not None:
        preds_top1 = df["index_top1"]
        correct_top1 = (preds_top1 == args.gt_index).sum()
        asr_top1 = correct_top1 / total * 100
        print(f"[ASR] Top-1 (index): {asr_top1:.2f}% ({correct_top1}/{total})")

        # Top-5 accuracy
        correct_top5 = 0
        matched_paths = []
        for _, row in df.iterrows():
            preds = [row[f"index_top{k}"] for k in range(1, 6)]
            if args.gt_index in preds:
                correct_top5 += 1
                matched_paths.append(row["path"])
        asr_top5 = correct_top5 / total * 100
        print(f"[ASR] Top-5 (index): {asr_top5:.2f}% ({correct_top5}/{total})")

        if matched_paths:
            print("Matched Top-5 images:")
            for p in matched_paths:
                print("  ", p)


if __name__ == "__main__":
    main()
