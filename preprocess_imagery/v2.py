import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def histograEqualColor(img_bgr):
    img_ycrcb=cv.cvtColor(img_bgr,cv.COLOR_BGR2YCrCb)
    y=img_ycrcb[:,:,0]

    hist=cv.calcHist([y],[0],None,[256],[0,256])
    cdf=hist.cumsum()
    cdfNorm=cdf*float(hist.max())/cdf.max()

    plt.figure()
    plt.subplot(231)
    plt.imshow(cv.cvtColor(img_bgr,cv.COLOR_BGR2RGB))
    plt.title("Oryginal")

    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdfNorm,color='b')

    y_eq=cv.equalizeHist(y)
    img_ycrcb_eq=img_ycrcb.copy()
    img_ycrcb_eq[:,:,0]=y_eq
    img_eq_bgr=cv.cvtColor(img_ycrcb_eq,cv.COLOR_YCrCb2BGR)

    equhist=cv.calcHist([y_eq],[0],None,[256],[0,256])
    eqycdf=equhist.cumsum()
    eqycdfNorm=eqycdf*float(equhist.max())/eqycdf.max()

    plt.subplot(232)
    plt.imshow(cv.cvtColor(img_eq_bgr,cv.COLOR_BGR2RGB))
    plt.title("Equalizacja")

    plt.subplot(235)
    plt.plot(equhist)
    plt.plot(eqycdfNorm,color='b')

    clahe=cv.createCLAHE(clipLimit=5,tileGridSize=(8,8))
    y_clahe=clahe.apply(y)

    img_ycrcb_clahe=img_ycrcb.copy()
    img_ycrcb_clahe[:,:,0]=y_clahe
    img_clahe_bgr=cv.cvtColor(img_ycrcb_clahe,cv.COLOR_YCrCb2BGR)

    clahehist=cv.calcHist([y_clahe],[0],None,[256],[0,256])
    clahecdf=clahehist.cumsum()
    clahecdfNorm=clahecdf*float(clahehist.max())/clahecdf.max()

    plt.subplot(233)
    plt.imshow(cv.cvtColor(img_clahe_bgr,cv.COLOR_BGR2RGB))
    plt.title("CLAHE")

    plt.subplot(236)
    plt.plot(clahehist)
    plt.plot(clahecdfNorm,color='b')    

    plt.show()


def unsharpMaskingColor(img_bgr, sigma=1.5, alpha=1.0):
    # to YCrCb
    img_ycrcb = cv.cvtColor(img_bgr, cv.COLOR_BGR2YCrCb)
    y = img_ycrcb[:,:,0]
    
    hist = cv.calcHist([y], [0], None, [256], [0,256])
    
    #gaussian blur
    # ksize must be odd,calculate it from sigma
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    y_blurred = cv.GaussianBlur(y, (ksize, ksize), sigma)
    
    #unsharp masking: I_sharp = I + alpha * (I - I_blurred)
    #cv.addWeighted for safe arithmetic with clipping
    y_sharp = cv.addWeighted(y, 1.0 + alpha, y_blurred, -alpha, 0)
    
    # alternative manual approach
    # y_sharp = y.astype(np.float32) + alpha * (y.astype(np.float32) - y_blurred.astype(np.float32))
    # y_sharp = np.clip(y_sharp, 0, 255).astype(np.uint8)
    
    img_ycrcb_sharp = img_ycrcb.copy()
    img_ycrcb_sharp[:,:,0] = y_sharp
    img_sharp_bgr = cv.cvtColor(img_ycrcb_sharp, cv.COLOR_YCrCb2BGR)
    
    sharp_hist = cv.calcHist([y_sharp], [0], None, [256], [0,256])
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(221)
    plt.imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
    plt.title("Oryginal")
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(cv.cvtColor(img_sharp_bgr, cv.COLOR_BGR2RGB))
    plt.title(f"Unsharp Masking (σ={sigma}, α={alpha})")
    plt.axis('off')
    
    plt.subplot(223)
    plt.plot(hist, label='Original', color='blue')
    plt.title("Histogram - Original")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(224)
    plt.plot(sharp_hist, label='Sharpened', color='red')
    plt.title("Histogram - Sharpened")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return img_sharp_bgr


def gammaCorrection(img_bgr, gamma=1.0):
    img_ycrcb = cv.cvtColor(img_bgr, cv.COLOR_BGR2YCrCb)
    y = img_ycrcb[:,:,0]
    
    hist = cv.calcHist([y], [0], None, [256], [0,256])
    
    #lookup table for gamma correction
    lookUpTable = np.array([((i / 255.0) ** (1.0/gamma)) * 255 
                            for i in range(256)]).astype(np.uint8)
    
    #LUT (works on all channels)
    img_gamma_bgr = cv.LUT(img_bgr, lookUpTable)
    
    img_gamma_ycrcb = cv.cvtColor(img_gamma_bgr, cv.COLOR_BGR2YCrCb)
    y_gamma = img_gamma_ycrcb[:,:,0]
    gamma_hist = cv.calcHist([y_gamma], [0], None, [256], [0,256])
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(221)
    plt.imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
    plt.title("Oryginal")
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(cv.cvtColor(img_gamma_bgr, cv.COLOR_BGR2RGB))
    plt.title(f"Gamma Correction (γ={gamma})")
    plt.axis('off')
    
    plt.subplot(223)
    plt.plot(hist, label='Original', color='blue')
    plt.title("Histogram - Original")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(224)
    plt.plot(gamma_hist, label=f'Gamma={gamma}', color='red')
    plt.title("Histogram - Gamma Corrected")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return img_gamma_bgr


def normalization(img_bgr, method='minmax', target_range=(0, 255)):
    img_ycrcb = cv.cvtColor(img_bgr, cv.COLOR_BGR2YCrCb)
    y = img_ycrcb[:,:,0]
    
    hist = cv.calcHist([y], [0], None, [256], [0,256])
    
    img_float = img_bgr.astype(np.float32)
    
    if method == 'minmax':
        # minmax: (x - min) / (max - min) * range + min_target
        img_min = img_float.min()
        img_max = img_float.max()
        img_norm = (img_float - img_min) / (img_max - img_min)
        img_norm = img_norm * (target_range[1] - target_range[0]) + target_range[0]
        
    elif method == 'zscore':
        # Z-score: (x - mean) / std, then scale to target_range
        mean = img_float.mean()
        std = img_float.std()
        img_norm = (img_float - mean) / (std + 1e-7)  # epsilon - for not deviding by zero
        # scale to target range (assume ±3 std covers most values)
        img_norm = np.clip(img_norm, -3, 3)  # clip outliers
        img_norm = (img_norm + 3) / 6  # normalize to [0, 1]
        img_norm = img_norm * (target_range[1] - target_range[0]) + target_range[0]
        
    elif method == 'percentile':
        #percentile clipping: clip to 2nd-98th percentile
        p2 = np.percentile(img_float, 2)
        p98 = np.percentile(img_float, 98)
        img_clipped = np.clip(img_float, p2, p98)
        img_norm = (img_clipped - p2) / (p98 - p2)
        img_norm = img_norm * (target_range[1] - target_range[0]) + target_range[0]
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'minmax', 'zscore', or 'percentile'")
    
    img_norm_bgr = np.clip(img_norm, target_range[0], target_range[1]).astype(np.uint8)
    
    img_norm_ycrcb = cv.cvtColor(img_norm_bgr, cv.COLOR_BGR2YCrCb)
    y_norm = img_norm_ycrcb[:,:,0]
    norm_hist = cv.calcHist([y_norm], [0], None, [256], [0,256])
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(221)
    plt.imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
    plt.title("Oryginal")
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(cv.cvtColor(img_norm_bgr, cv.COLOR_BGR2RGB))
    plt.title(f"Normalization ({method})")
    plt.axis('off')
    
    plt.subplot(223)
    plt.plot(hist, label='Original', color='blue')
    plt.title("Histogram - Original")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(224)
    plt.plot(norm_hist, label=f'{method}', color='red')
    plt.title("Histogram - Normalized")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return img_norm_bgr


def save_img(img, img_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    name_img = os.path.splitext(os.path.basename(img_path))[0] + ".png"
    save_path = os.path.join(output_dir, name_img)
    cv.imwrite(save_path, img)
    print(f"saved: {save_path}")


if __name__ == "__main__":
    root = os.getcwd()
    imgpath = os.path.join(root,"datasets/UAV_VisLoc_dataset/02/drone/02_0021.JPG")
    img = cv.imread(imgpath)
    output_dir = os.path.join(root, "output_processed")
    os.makedirs(output_dir, exist_ok=True)

    print("Testing Gamma Correction...")
    img_gamma = gammaCorrection(img, gamma=0.9)  # brighten shadows
    save_img(img_gamma, imgpath.replace(".JPG","_gamma.JPG"), output_dir)
    img_gamma_dark = gammaCorrection(img, gamma=1.2)  # darken highlights
    save_img(img_gamma_dark, imgpath.replace(".JPG","_gamma_dark.JPG"), output_dir)
    
    print("\nminmax normalization")
    img_norm_minmax = normalization(img, method='minmax')
    save_img(img_norm_minmax, imgpath.replace(".JPG","_norm_minmax.JPG"), output_dir)

    print("\nz-score")
    img_norm_zscore = normalization(img, method='zscore')
    save_img(img_norm_zscore, imgpath.replace(".JPG","_norm_zscore.JPG"), output_dir)
    
    print("\npercentile")
    img_norm_percentile = normalization(img, method='percentile')
    save_img(img_norm_percentile, imgpath.replace(".JPG","_norm_percentile.JPG"), output_dir)
    
    #or if need 0-1 range
    # img_norm = normalization(img, method='minmax', target_range=(0, 1))