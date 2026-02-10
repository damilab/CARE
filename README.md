<p align="center">

<h2 align="center"><strong> Co-occurring Associated REtained concepts in Diffusion Unlearning</strong></h2>
<h3 align="center"><strong>[ICLR 2026 Accepted]</strong></h3>
</p>

<p align="center">
  <a href="https://openreview.net/forum?id=Ryc7jKP6H9&noteId=LjNdgM6vZg">📄 Paper (OpenReview)</a>
</p>

<p align="center">
  <img src="assets/overview.png" width="100%">
</p>


## 🔍 Overview
This repository provides the **official implementation of ReCARE**, a diffusion model unlearning framework that **removes a target concept while explicitly preserving benign co-occurring concepts (CARE)**.

Existing unlearning methods often suppress benign concepts that naturally co-occur with the erase target (e.g., erasing *nudity* unintentionally removes *person*).

**ReCARE** addresses this issue by:
- **Automatically constructing a CARE-set** from target images
- **Integrating the CARE-set into unlearning training** to preserve co-occurring semantics

## Instructions for Code Usage
### 🛠️ Environment Setup

```shell
git clone https://github.com/damilab/CARE.git
cd CARE

# Linux (recommended)
conda env create -f environment.yaml
conda activate recare

# clip
pip install git+https://github.com/openai/CLIP.git
```

### 🎨 Image Generation
Generate images containing the erase target. These images are later used both for CARE-set construction and unlearning training.

```shell
# Object Concept (Nudity)
python generate_images.py --output_dir data/nudity/images/ --prompt "A photo of a nude person" --num_images 500

# Style Concept (Van Gogh)
python generate_images.py --output_dir data/vangogh/images/ --prompt "A painting in the style of Van Gogh" --num_images 500
```
### 📚 CARE Dictionary Construction
Construct the CARE-set, a curated vocabulary of benign co-occurring concepts.

```shell
# Object Concept (Nudity)
python utils/dictionary.py --reference_word 'nudity' --image_dir data/nudity/images/ --output_dir data/nudity/ --prompt_style photo

# Style Concept (Van Gogh)
python utils/dictionary.py --reference_word 'Van Gogh' --image_dir data/vangogh/images/ --output_dir data/vangogh/ --prompt_style style
```

This step produces a ```careset.json``` file, which serves as the CARE-set anchor during training.

### 🚀 Training
```shell
# Object Concept (Nudity)
python -W ignore train.py --erase_concept 'nudity' --train_method noxattn --train_data_dir data/nudity/images/ --learnable_property 'object' --initializer_token 'person' --output_dir recare_weights/nudity --compositional_guidance_scale 2 --n_iterations 2 --num_of_adv_concepts 2 --anchor_concept_path data/nudity/careset.json

# Style Concept (Van Gogh)
python -W ignore train.py --erase_concept 'Van Gogh' --train_method noxattn --train_data_dir data/vangogh/images/ --learnable_property 'style' --initializer_token 'art' --output_dir recare_weights/vangogh --compositional_guidance_scale 2 --n_iterations 2 --num_of_adv_concepts 2 --anchor_concept_path data/vangogh/careset.json
```

### 📊 CARE Evaluation
Evaluate whether benign co-occurring concepts are preserved after unlearning, using the proposed CARE score.
```shell
python metrics/care_eval.py \
  --gen_prompts_file prompts_person.txt \
  --targets 'person' \
  --num_images 10 \
  --unet_checkpoint recare_weights/nudity/ReCARE-Diffusers-UNet.pt \
  --out_dir care_outputs \
  --save_images
```
Additional evaluation scripts (ASR, COCO-based FID, CLIP score) are available in ```metrics/```. \
For style-based evaluation, download the pretrained style classifier checkpoint and place it at: ```metrics/style_classifier/checkpoint-2800```


## 💾 Pretrained Checkpoints
Pretrained ReCARE model checkpoints (`ReCARE-Diffusers-UNet.pt`) are provided via [Google Drive](https://drive.google.com/drive/folders/1p7EeyE2RuvcyyySxpjNY0sc6t7lPsUf7?usp=drive_link). After downloading, place them under the `recare_weights/` directory (e.g., `recare_weights/nudity/`).
