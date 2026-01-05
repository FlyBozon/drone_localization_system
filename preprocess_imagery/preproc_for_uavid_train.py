import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def downscale_image_and_mask(image_path, mask_path, target_size=(1024, 512)):
    image = cv.imread(image_path)
    mask = cv.imread(mask_path, cv.IMREAD_UNCHANGED)
    image_resized = cv.resize(image, target_size, interpolation=cv.INTER_AREA)
    mask_resized = cv.resize(mask, target_size, interpolation=cv.INTER_NEAREST)
    return image_resized, mask_resized

#combine effects from effects.py into one function
def random_degrade(img):
    if random.random() < 0.5:
        img = cv.GaussianBlur(img, (random.choice([5, 9, 13]),) * 2, 0)
    if random.random() < 0.5:
        size = random.choice([10,15,20])
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = 1.0
        kernel /= size
        img = cv.filter2D(img, -1, kernel)
    if random.random() < 0.5:
        sigma = random.randint(10,40)
        noise = np.random.randn(*img.shape) * sigma
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.5:
        quality = random.randint(5,25)
        _, enc = cv.imencode('.jpg', img, [int(cv.IMWRITE_JPEG_QUALITY), quality])
        img = cv.imdecode(enc, cv.IMREAD_COLOR)
    if random.random() < 0.5:
        scale = random.uniform(0.3,0.7)
        h, w = img.shape[:2]
        small = cv.resize(img, (int(w*scale), int(h*scale)), cv.INTER_AREA)
        img = cv.resize(small, (w, h), cv.INTER_LINEAR)
    if random.random() < 0.5:
        factor = random.uniform(0.4,1.2)
        img = np.clip(img * factor, 0, 255).astype(np.uint8)
    return img

def show_blend(image, mask, alpha=0.4):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    mask = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
    img_f = image.astype(float) / 255.0
    mask_f = mask.astype(float) / 255.0
    mask_bool = np.any(mask > 0, axis=2, keepdims=True)
    blended = img_f.copy()
    blended[mask_bool.squeeze()] = (1 - alpha) * img_f[mask_bool.squeeze()] + alpha * mask_f[mask_bool.squeeze()]
    return blended

root = "datasets/UAVid/uavid_test"
out_root = "output/dataset/uavid"
# root = "/data/markryku/datasets/uavid"
# out_root = "/data/markryku/dataset/uavid"

def process_dataset(root, out_root, with_labels=True):
    pairs = []
    for path, dirs, files in os.walk(root):
        if path.endswith("Images"):
            mask_dir = os.path.join(os.path.dirname(path), "Labels")
            if with_labels and os.path.isdir(mask_dir):
                for f in files:
                    if f.lower().endswith(".png"):
                        img_path = os.path.join(path, f)
                        mask_path = os.path.join(mask_dir, f)
                        if os.path.exists(mask_path):
                            pairs.append((img_path, mask_path))
                        else:
                            print(f"Mask does not exist: {mask_path}")
            else:
                for f in files:
                    if f.lower().endswith(".png"):
                        img_path = os.path.join(path, f)
                        pairs.append((img_path, None))
    out_img_dir = os.path.join(out_root, os.path.basename(root), "Images")
    out_mask_dir = os.path.join(out_root, os.path.basename(root), "Labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    for img_path, mask_path in pairs:
        if mask_path:
            img_res, mask_res = downscale_image_and_mask(img_path, mask_path)
            if img_res is None or (mask_path and mask_res is None):
                continue
            #good img
            cv.imwrite(os.path.join(out_img_dir, os.path.basename(img_path)), img_res)
            cv.imwrite(os.path.join(out_mask_dir, os.path.basename(mask_path)), mask_res)
            #1-2 degraded versions
            for i in range(random.randint(1,3)):
                bad_img = random_degrade(img_res.copy())
                name_bad = os.path.splitext(os.path.basename(img_path))[0] + f"_bad{i}.png"
                mask_bad = os.path.splitext(os.path.basename(mask_path))[0] + f"_bad{i}.png"
                cv.imwrite(os.path.join(out_img_dir, name_bad), bad_img)
                cv.imwrite(os.path.join(out_mask_dir, mask_bad), mask_res)
        else:
            img = cv.imread(img_path)
            if img is None:
                continue
            img_res = cv.resize(img, (1024, 512), interpolation=cv.INTER_AREA)
            cv.imwrite(os.path.join(out_img_dir, os.path.basename(img_path)), img_res)
            for i in range(random.randint(1,3)):
                bad_img = random_degrade(img_res.copy())
                name_bad = os.path.splitext(os.path.basename(img_path))[0] + f"_bad{i}.png"
                cv.imwrite(os.path.join(out_img_dir, name_bad), bad_img)

process_dataset(root, out_root, with_labels=True)

good_images = []
bad_images = []

for seq in os.listdir(out_root):
    img_dir = os.path.join(out_root, seq, "Images")
    mask_dir = os.path.join(out_root, seq, "Labels")
    if os.path.isdir(img_dir):
        for f in os.listdir(img_dir):
            good_images.append((os.path.join(img_dir,f), os.path.join(mask_dir,f)))
    if os.path.isdir(img_dir):
        for f in os.listdir(img_dir):
            bad_images.append((os.path.join(img_dir,f), os.path.join(mask_dir,f)))

samples = random.sample(good_images + bad_images, min(4,len(good_images + bad_images)))

plt.figure(figsize=(12,12))
for i, (img_path, mask_path) in enumerate(samples):
    img = cv.imread(img_path)
    mask = cv.imread(mask_path)
    if img is None or mask is None:
        continue
    plt.subplot(2,2,i+1)
    blended = show_blend(img, mask)
    plt.imshow(blended)
    plt.axis("off")
plt.tight_layout()
plt.show()