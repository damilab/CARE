import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import json
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.utils import *
from utils.dataset import TextualInversionDataset
from utils.apg import *
import warnings
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

def train_erasing(erase_concept, erase_from, train_method, iterations, negative_guidance, lr, save_path, diffuser, device):
    seed = 42
    np.random.seed(seed)      
    random.seed(seed)        
    torch.manual_seed(seed) 

    nsteps = 50

    diffuser.requires_grad = True
    diffuser.train()

    finetuner = FineTunedModel(diffuser, train_method=train_method)

    optimizer = torch.optim.AdamW(finetuner.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()

    pbar = tqdm(range(iterations))
    erase_concept = [a.strip() for a in erase_concept.split(',')]
    erase_from = [a.strip() for a in erase_from.split(',')]

    if len(erase_from) != len(erase_concept):
        if len(erase_from) == 1:
            erase_from = [erase_from[0]] * len(erase_concept)
        else:
            raise ValueError("Erase concepts and target concepts must have matching lengths.")

    erase_concept = [[e, f] for e, f in zip(erase_concept, erase_from)]

    torch.cuda.empty_cache()

    for i in pbar:
        with torch.no_grad():
            index = np.random.choice(len(erase_concept), 1, replace=False)[0]
            erase_concept_sampled = erase_concept[index]

            neutral_text_embeddings = diffuser.get_text_embeddings([''], n_imgs=1)
            positive_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[0]], n_imgs=1)
            target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled[1]], n_imgs=1)

            diffuser.set_scheduler_timesteps(nsteps)
            optimizer.zero_grad()
            iteration = torch.randint(1, nsteps - 1, (1,)).item()
            latents = diffuser.get_initial_latents(1, 512, 1)

            with finetuner:
                latents_steps, _ = diffuser.diffusion(
                    latents,
                    positive_text_embeddings,
                    start_iteration=0,
                    end_iteration=iteration,
                    guidance_scale=3, 
                    show_progress=False
                )

            diffuser.set_scheduler_timesteps(1000)
            iteration = int(iteration / nsteps * 1000)
            positive_latents = diffuser.predict_noise(iteration, latents_steps[0], positive_text_embeddings, guidance_scale=1)
            neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)
            target_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)

            torch.cuda.empty_cache()

            if erase_concept_sampled[0] == erase_concept_sampled[1]:
                target_latents = neutral_latents.clone().detach()
        
        with finetuner:
            negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
     
        pred_neg_guidance = normalized_guidance(positive_latents, neutral_latents, negative_guidance)
        loss = criteria(negative_latents, pred_neg_guidance)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        pbar.set_description(f"Loss: {loss.item():.4f}")

    with finetuner:
        torch.save(diffuser.unet.state_dict(), save_path)

    del neutral_text_embeddings, positive_text_embeddings, target_text_embeddings
    del latents, latents_steps, positive_latents, neutral_latents, target_latents, negative_latents
    del loss, optimizer, finetuner
    del erase_concept, erase_from, criteria, pbar, index, erase_concept_sampled, iteration, nsteps

    torch.cuda.empty_cache()
    diffuser.eval()
    return diffuser


def train_concept_inversion(
    diffuser,
    placeholder_token, 
    initializer_token, 
    train_data_dir, 
    lr, 
    save_path, 
    device, 
    num_vectors=1, 
    max_train_steps=3000,  
    resolution=512, 
    learnable_property="object",
    lr_scheduler="constant", 
    lr_warmup_steps=0, 
    scale_lr=False,  
    iteration=None,
    num_iterations=None,
    center_crop=False
):
    
    
    seed = 42
    np.random.seed(seed)     
    random.seed(seed)        
    torch.manual_seed(seed) 

    diffuser.requires_grad = False

    for param in diffuser.text_encoder.text_model.embeddings.token_embedding.parameters():
        param.requires_grad = True

    tokenizer = diffuser.tokenizer

    placeholder_tokens = [placeholder_token]
    additional_tokens = [f"{placeholder_token}_{i}" for i in range(1, num_vectors)]
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != num_vectors:
        raise ValueError(f"Token '{placeholder_token}' already exists in tokenizer. Use a different token name.")

    initializer_token_id = tokenizer.convert_tokens_to_ids([initializer_token])[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    diffuser.text_encoder.resize_token_embeddings(len(tokenizer))

    with torch.no_grad():
        token_embeds = diffuser.text_encoder.get_input_embeddings().weight.data
        ctr = 0
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
            ctr += 1
        print(f"Initialized {ctr} placeholder token embeddings with '{initializer_token}' token embeddings.")

    org_token_embeds = diffuser.text_encoder.get_input_embeddings().weight.data.clone()
    

    dataset = TextualInversionDataset(
        data_root=train_data_dir,
        tokenizer=tokenizer,
        size=resolution,
        placeholder_token=" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids)),
        repeats=100,
        set="train",
        learnable_property=learnable_property,
        center_crop=center_crop,
        iteration=iteration,
        num_iterations=num_iterations
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    steps_per_epoch = len(dataloader)
    num_train_epochs = math.ceil(max_train_steps / steps_per_epoch)

    if scale_lr:
        effective_batch_size = dataloader.batch_size
        lr *= effective_batch_size 

    optimizer = torch.optim.AdamW(diffuser.text_encoder.get_input_embeddings().parameters(), lr=lr)
    scheduler = get_scheduler(lr_scheduler, optimizer, num_warmup_steps=lr_warmup_steps, num_training_steps=max_train_steps)

    progress_bar = tqdm(total=max_train_steps, desc="Textual Inversion Progress", unit="step")
    global_step = 0

    for epoch in range(num_train_epochs):
        diffuser.text_encoder.train()
        
        for step, batch in enumerate(dataloader):
            if global_step >= max_train_steps:
                break

            optimizer.zero_grad()

            latents = diffuser.vae.encode(batch["pixel_values"].to(device)).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 999, (latents.shape[0],), device=latents.device)
            noisy_latents = diffuser.scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = diffuser.text_encoder(batch["input_ids"].to(device)).last_hidden_state
            model_pred = diffuser.unet(noisy_latents, timesteps, encoder_hidden_states)["sample"]

            target = noise
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            scheduler.step()

            index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool, device=device)
            index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False  # False indicates trainable embeddings

            with torch.no_grad():
                diffuser.text_encoder.get_input_embeddings().weight.data[index_no_updates] = org_token_embeds[index_no_updates]
            
            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)
            global_step += 1
            if global_step >= max_train_steps:
                break

    progress_bar.close()

    if not (save_path == None):
        torch.save(diffuser.text_encoder.state_dict(), save_path)
    else:
        print("Not saving the text encoder state dict as save_path is None.")

    del optimizer, scheduler, dataset, dataloader, progress_bar, global_step, steps_per_epoch, num_train_epochs, effective_batch_size, token_embeds, index_no_updates, model_pred, target, loss, batch, latents, noise, timesteps, noisy_latents, encoder_hidden_states, placeholder_tokens, additional_tokens, initializer_token_id, placeholder_token_ids, tokenizer, org_token_embeds
    torch.cuda.empty_cache()
    diffuser.eval()
    return diffuser


def iterative_textual_inversion(
    diffuser,
    initial_erase_concept,
    initializer_token,
    train_data_dir,
    train_method,
    lr,
    ti_lr,
    negative_guidance,
    iterations,
    n_iterations,
    device,
    ti_max_train_steps,
    learnable_property,
    output_dir, 
    generic_prompt,
    center_crop=False
):
    current_concept = initial_erase_concept
    saved_tokens = {}

    os.makedirs(output_dir, exist_ok=True)
    
    for iteration in range(n_iterations):
        placeholder_token = generate_unique_placeholder_token(saved_tokens, iteration)
        saved_tokens[f'{iteration}'] = placeholder_token

        erased_weights_path = os.path.join(output_dir, f"erased_unet_iteration_{iteration}.pt")
        ti_encoder_path = os.path.join(output_dir, f"ti_text_encoder_iteration_{iteration}.pt")
        
        print(f"\n---------------------------------- Iteration {iteration + 1}/{n_iterations} ----------------------------------")
        print(f"Erasing concept: {current_concept} -> Placeholder token: '{placeholder_token}' (initialized from '{initializer_token}')")

        diffuser = train_erasing(
            erase_concept=current_concept,
            erase_from=current_concept,
            train_method=train_method,
            iterations=iterations,
            negative_guidance=negative_guidance,
            lr=lr,
            save_path=erased_weights_path,
            diffuser=diffuser,
            device=device
        )
        print(f"Erased weights saved to {erased_weights_path}")

        diffuser.unet.load_state_dict(torch.load(erased_weights_path))
        torch.cuda.empty_cache()

        diffuser = train_concept_inversion(
            diffuser=diffuser,
            placeholder_token=placeholder_token,
            initializer_token=initializer_token,
            train_data_dir=train_data_dir,
            lr=ti_lr,
            save_path=ti_encoder_path,
            device=device,
            max_train_steps=ti_max_train_steps,
            learnable_property=learnable_property,
            scale_lr=True, 
            iteration=iteration,
            num_iterations=n_iterations,
            center_crop=center_crop
        )
        print(f"Text encoder with placeholder '{placeholder_token}' saved to {ti_encoder_path}")

        current_concept = placeholder_token

        print(f"---------------------------------- Iteration {iteration + 1}/{n_iterations} complete. ----------------------------------\n\n")
        torch.cuda.empty_cache()


    final_model_path = os.path.join(output_dir, "recare_stage1.pt")
    torch.save({
        'model_state_dict': diffuser.state_dict(),
        'saved_tokens': saved_tokens
    }, final_model_path)
    print(f"\nIterative stage complete. Final model saved to {final_model_path}")
    print(f"Placeholder tokens: {saved_tokens}")

    return final_model_path, saved_tokens, diffuser


def robust_erase_for_care(erase_concepts, train_method, iterations, compositional_guidance_scale, lr, save_path, diffuser, anchor_concepts_path):
    nsteps = 50

    with open(anchor_concepts_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    anchor_keys = []
    retain_keys = []

    for k, v in data.items():
        if not isinstance(v, list):
            continue

        k_low = k.lower()
        if ("photo" in k_low) or ("painting" in k_low):
            retain_keys.append(k)
        else:
            anchor_keys.append(k)

    if len(retain_keys) != 1:
        raise KeyError(f"[CARE JSON] Expected exactly 1 retain key containing 'photo' or 'painting', got: {retain_keys}")
    if len(anchor_keys) != 1:
        raise KeyError(f"[CARE JSON] Expected exactly 1 anchor key (non-photo/painting), got: {anchor_keys}")

    retain_key = retain_keys[0]
    anchor_key = anchor_keys[0]

    all_anchor_concepts = data[anchor_key]
    all_retain_concepts = data[retain_key]

    diffuser.requires_grad = True
    diffuser.train()

    finetuner = FineTunedModel(diffuser, train_method=train_method)

    optimizer = torch.optim.AdamW(finetuner.parameters(), lr=lr)
    criteria = torch.nn.MSELoss()
    
    torch.cuda.empty_cache()

    # ------- erase loss에 쓰이는 care concepts ------
    total_sentences = len(all_anchor_concepts)
    appearances_per_sentence = iterations // total_sentences
    balanced_list = all_anchor_concepts * appearances_per_sentence
    np.random.shuffle(balanced_list)
    remainder = iterations - len(balanced_list)
    if remainder > 0:
        balanced_list.extend(np.random.choice(all_anchor_concepts, remainder, replace=False))

    # ------- retain loss에 쓰이는 care concepts ------
    total_retain_sentences = len(all_retain_concepts)
    appearances_per_sentence = iterations // total_retain_sentences
    balanced_retain_list = all_retain_concepts * appearances_per_sentence
    np.random.shuffle(balanced_retain_list)
    remainder = iterations - len(balanced_retain_list)
    if remainder > 0:
        balanced_retain_list.extend(np.random.choice(all_retain_concepts, remainder, replace=False))

    # ------ erase concepts -------
    total_concepts = len(erase_concepts)
    appearances_per_concept = iterations // total_concepts
    balanced_erase_list = erase_concepts * appearances_per_concept
    np.random.shuffle(balanced_erase_list)
    remainder = iterations - len(balanced_erase_list)
    if remainder > 0:
        balanced_erase_list.extend(np.random.choice(erase_concepts, remainder, replace=False))

    pbar = tqdm(range(iterations))
    for i in pbar:
        with torch.no_grad():
            erase_concept_sampled = balanced_erase_list[i]
            anchor_concepts = [balanced_list[i]]
            retain_concepts = [balanced_retain_list[i]]

            print(f"Erasing concept: {erase_concept_sampled} from anchor concept: {anchor_concepts} with retain concept {retain_concepts} at iteration {i}")

            neutral_text_embeddings = diffuser.get_text_embeddings([''], n_imgs=1)
            target_text_embeddings = diffuser.get_text_embeddings([erase_concept_sampled], n_imgs=1)
            anchor_text_embeddings = diffuser.get_text_embeddings(anchor_concepts, n_imgs=1)
            retain_text_embeddings = diffuser.get_text_embeddings(retain_concepts, n_imgs=1)

            negative_word_embs = []
            for neg_word in erase_concepts:
                negative_word_embs.append(diffuser.get_text_embeddings([neg_word], n_imgs=1))

            anchor_word_embs = []
            for anchor_word in anchor_concepts:
                anchor_word_embs.append(diffuser.get_text_embeddings([anchor_word], n_imgs=1))
            
            retain_word_embs = []
            for retain_word in retain_concepts:
                retain_word_embs.append(diffuser.get_text_embeddings([retain_word], n_imgs=1))

            diffuser.set_scheduler_timesteps(nsteps)
            optimizer.zero_grad()
            iteration = torch.randint(1, nsteps - 1, (1,)).item()
            latents = diffuser.get_initial_latents(1, 512, 1)

            with finetuner:
                latents_steps, _ = diffuser.diffusion(
                    latents,
                    target_text_embeddings,
                    start_iteration=0,
                    end_iteration=iteration,
                    guidance_scale=3, 
                    show_progress=False
                )


            diffuser.set_scheduler_timesteps(1000)
            iteration = int(iteration / nsteps * 1000)
            neutral_latents = diffuser.predict_noise(iteration, latents_steps[0], neutral_text_embeddings, guidance_scale=1)
            target_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)

            e_negatives_latents = []
            for emb_neg in negative_word_embs:
                e_negatives_latents.append(diffuser.predict_noise(iteration, latents_steps[0], emb_neg, guidance_scale=1))

            e_anchor_latents = []
            for emb_anchor in anchor_word_embs:
                e_anchor_latents.append(diffuser.predict_noise(iteration, latents_steps[0], emb_anchor, guidance_scale=1))
            
            e_retain_latents = []
            for emb_retain in retain_word_embs:
                e_retain_latents.append(diffuser.predict_noise(iteration, latents_steps[0], emb_retain, guidance_scale=1))\

            torch.cuda.empty_cache()
        
        with finetuner:
            negative_latents = diffuser.predict_noise(iteration, latents_steps[0], target_text_embeddings, guidance_scale=1)
            retain_latents = diffuser.predict_noise(iteration, latents_steps[0], retain_text_embeddings, guidance_scale=1)

        neg_guidance_scales = []
        for _ in range(len(e_negatives_latents)):
            neg_guidance_scales.append(-(float(compositional_guidance_scale)))

        pos_guidance_scales = []
        for _ in range(len(e_anchor_latents)):
            pos_guidance_scales.append(float(compositional_guidance_scale))

        combined_conditions = e_negatives_latents + e_anchor_latents
        combined_guidance_scales = neg_guidance_scales + pos_guidance_scales

        compositional_guidance_estimate = normalized_compositional_guidance(combined_conditions, neutral_latents, combined_guidance_scales)
        erase_loss = criteria(negative_latents, compositional_guidance_estimate)
        retain_loss = criteria(retain_latents, e_retain_latents[0])
        # total loss
        loss = erase_loss + 1*retain_loss

        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        pbar.set_description(f"Total Loss: {loss.item():.4f} | Erase: {erase_loss.item():.4f} | Retain: {retain_loss.item():.4f}")

    with finetuner:
        torch.save(diffuser.unet.state_dict(), save_path)

    del neutral_text_embeddings, target_text_embeddings, anchor_text_embeddings, retain_text_embeddings
    del latents, latents_steps, neutral_latents, target_latents, negative_latents, retain_latents
    del loss, optimizer, finetuner
    del criteria, pbar, erase_concept_sampled, iteration, nsteps, e_negatives_latents, e_anchor_latents, e_retain_latents
    del negative_word_embs, anchor_word_embs, retain_word_embs

    torch.cuda.empty_cache()


def recare(args):
    diffuser = StableDiffuser(scheduler='DDIM').to(args.device)

    iti_start_time = time.time()
    print(f"===================================== Iterative Textual Inversion =====================================")
    final_model_path, saved_tokens, diffuser = iterative_textual_inversion(
        diffuser=diffuser,
        initial_erase_concept=args.erase_concept,
        initializer_token=args.initializer_token,
        train_data_dir=args.train_data_dir,
        train_method=args.train_method,
        lr=args.recare_stage1_lr,
        ti_lr=args.ti_lr,
        negative_guidance=args.negative_guidance,
        iterations=args.iterations,
        n_iterations=args.n_iterations,
        device=args.device,
        ti_max_train_steps=args.ti_max_train_steps,
        learnable_property=args.learnable_property,
        output_dir=args.output_dir,
        generic_prompt=args.generic_prompt,
        center_crop=args.center_crop
    )
    iti_end_time = time.time()
    print(f"iterative_textual_inversion : {iti_end_time - iti_start_time} seconds\n")

    del diffuser
    torch.cuda.empty_cache()

    print(f"===================================== Robust Erase For CARE =====================================")
    diffuser = StableDiffuser(scheduler='DDIM').to(args.device)
    diffuser_copy = StableDiffuser(scheduler='DDIM').to(args.device)
    final_unet_path = os.path.join(args.output_dir, "ReCARE-Diffusers-UNet.pt")
    final_model_path = os.path.join(args.output_dir, "recare_stage1.pt")

    ckpt = torch.load(final_model_path)
    saved_tokens = ckpt['saved_tokens']
    for token in list(saved_tokens.values()):
        if token not in diffuser.tokenizer.get_vocab():
            print(f"!!!! Adding placeholder token '{token}' to tokenizer.")
            diffuser.tokenizer.add_tokens([token])
            diffuser_copy.tokenizer.add_tokens([token])
            diffuser.text_encoder.resize_token_embeddings(len(diffuser.tokenizer))
            diffuser_copy.text_encoder.resize_token_embeddings(len(diffuser.tokenizer))
    
    diffuser_copy.load_state_dict(ckpt['model_state_dict'])
    diffuser.text_encoder.load_state_dict(diffuser_copy.text_encoder.state_dict())
    del ckpt, diffuser_copy
    torch.cuda.empty_cache()

    final_concepts_to_erase = [args.erase_concept]
    adv_tokens_from_iti = list(saved_tokens.values())
    # print(f"erasing concepts found from ITI stage: {adv_tokens_from_iti}")
    if not args.num_of_adv_concepts == 0:
        adv_tokens_from_iti = adv_tokens_from_iti[0:args.num_of_adv_concepts]
        final_concepts_to_erase.extend(adv_tokens_from_iti)
    
    recare_start_time = time.time()
    robust_erase_for_care(
        erase_concepts=final_concepts_to_erase,
        train_method=args.train_method,
        iterations=args.iterations,
        compositional_guidance_scale=args.compositional_guidance_scale,
        lr=args.recare_stage2_lr,
        save_path=final_unet_path,
        diffuser=diffuser,
        anchor_concepts_path=args.anchor_concept_path
    )
    recare_end_time = time.time()
    del diffuser, saved_tokens
    torch.cuda.empty_cache()
    print(f"Final ReCARE unet saved to {final_unet_path}")
    print(f"===================================== Robust Erase For CARE : {recare_end_time - recare_start_time} seconds =====================================")
