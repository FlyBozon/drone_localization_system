import os
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

def degrade_blur(img, k=9):
    return cv.GaussianBlur(img, (k, k), 0)

#more realistic blur, as drones are dont have ideal blur, rather directional (pitch, row, yaw?), motion/
def degrade_drone_motion_blur(img, length=20, angle=5):
    kernel = np.zeros((length, length))
    center = length // 2
    for i in range(length):
        x = int(center + (i - center) * math.cos(math.radians(angle)))
        y = int(center + (i - center) * math.sin(math.radians(angle)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    kernel /= kernel.sum()
    return cv.filter2D(img, -1, kernel)


def degrade_motion_blur(img, size=15):
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = 1.0
    kernel /= size
    return cv.filter2D(img, -1, kernel)

def degrade_down_up(img, scale=0.5):
    h, w = img.shape[:2]
    small = cv.resize(img, (int(w*scale), int(h*scale)), cv.INTER_AREA)
    return cv.resize(small, (w, h), cv.INTER_LINEAR)

def degrade_noise(img, sigma=25):
    noise = np.random.randn(*img.shape) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

#more sensor like noise - iso, gain, electronic noise
def degrade_shot_noise(img, intensity=0.05):
    img_f = img.astype(np.float32) / 255.0
    noisy = np.random.poisson(img_f * intensity * 255) / (intensity * 255)
    noisy = np.clip(noisy, 0, 1)
    return (noisy * 255).astype(np.uint8)

def degrade_vignette(img, strength=0.7):
    rows, cols = img.shape[:2]
    kernel_x = cv.getGaussianKernel(cols, cols * strength)
    kernel_y = cv.getGaussianKernel(rows, rows * strength)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    return (img * mask[..., None]).astype(np.uint8)

def degrade_chromatic_aberration(img, shift=2):
    b,g,r = cv.split(img)
    b = np.roll(b, shift, axis=1)
    r = np.roll(r, -shift, axis=1)
    return cv.merge((b,g,r))

def degrade_brightness(img, factor=0.5):
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def degrade_jpeg(img, quality=10):
    _, enc = cv.imencode('.jpg', img, [int(cv.IMWRITE_JPEG_QUALITY), quality])
    return cv.imdecode(enc, cv.IMREAD_COLOR)

def degrade_lens_distortion(img, k1=-0.25, k2=0.05, p1=0.0, p2=0.0, k3=0.0):
    h, w = img.shape[:2]
    #lets assume approximate focal length
    fx = fy = 0.8 * w  
    cx, cy = w / 2, h / 2

    dist_coeffs = np.array([k1, k2, p1, p2, k3])
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0,  0,  1]])

    map1, map2 = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs,
                                            None, camera_matrix,
                                            (w, h),
                                            cv.CV_32FC1)
    distorted = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
    return distorted


def demo_degradations(img):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return {
        "original": img_rgb,
        "blur": cv.cvtColor(degrade_blur(img), cv.COLOR_BGR2RGB),
        "motion": cv.cvtColor(degrade_motion_blur(img), cv.COLOR_BGR2RGB),
        "down_up": cv.cvtColor(degrade_down_up(img), cv.COLOR_BGR2RGB),
        "noise": cv.cvtColor(degrade_noise(img), cv.COLOR_BGR2RGB),
        "brightness": cv.cvtColor(degrade_brightness(img), cv.COLOR_BGR2RGB),
        "jpeg": cv.cvtColor(degrade_jpeg(img), cv.COLOR_BGR2RGB),
        "lens_distortion": cv.cvtColor(degrade_lens_distortion(img), cv.COLOR_BGR2RGB),
        "motion_drone": cv.cvtColor(degrade_drone_motion_blur(img), cv.COLOR_BGR2RGB),
        "noise_shot": cv.cvtColor(degrade_shot_noise(img), cv.COLOR_BGR2RGB),
        "vignette": cv.cvtColor(degrade_vignette(img), cv.COLOR_BGR2RGB),
        "chromatic_aberration": cv.cvtColor(degrade_chromatic_aberration(img), cv.COLOR_BGR2RGB),
    }

def plot_effects(img):
    effects = demo_degradations(img)
    names = list(effects.keys())
    n = len(names)
    cols = 4
    rows = math.ceil(n / cols)

    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, name in enumerate(names):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(effects[name])
        plt.title(name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

root = "output/dataset/uavid/uavid_train/Images"

all_imgs = []
for path, _, files in os.walk(root):
    for f in files:
        if f.lower().endswith(".png"):
            all_imgs.append(os.path.join(path, f))

if len(all_imgs) == 0:
    print("No resized images found.")
    exit()

samples = random.sample(all_imgs, min(3, len(all_imgs)))

for img_path in samples:
    print("teeeeestt:", img_path)
    img = cv.imread(img_path)
    plot_effects(img)
