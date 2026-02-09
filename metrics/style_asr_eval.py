import os
from PIL import Image
from transformers import pipeline

def init_classifier(device, path):
    return pipeline('image-classification', model=path, device=device)

def style_eval(classifier, img):
    return classifier(img, top_k=129)

device = 0
input_dir = "/eval" 
target_artist = "vincent-van-gogh"

top_k = 3
classifier_path = "./style_classifier/checkpoint-2800"

classifier = init_classifier(device, classifier_path)

image_extensions = ['.jpg', '.jpeg', '.png']
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(tuple(image_extensions))]
print(f"\nFound {len(image_files)} images in total.\n")

accuracies = []
for i, image_file in enumerate(sorted(image_files)):
    image_path = os.path.join(input_dir, image_file)
    try:
        image = Image.open(image_path).convert("RGB")
    except:
        print(f"[{i+1}] {image_file}: ❌ Failed to load image")
        continue

    results = style_eval(classifier, image)[:top_k]
    top_labels = [r['label'] for r in results]
    success = target_artist in top_labels
    accuracies.append(success)
    
    status = "✅" if success else "❌"
    print(f"[{i+1:03}] {image_file}: {status} → ~ {top_k}: {top_labels[0:3]}")

accuracy = 100 * sum(accuracies) / len(accuracies)
print(f"\n🎯 {top_k} Accuracy for [{target_artist}] = {accuracy:.2f}%")
