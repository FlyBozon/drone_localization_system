#oblique to nadir
import cv2
import numpy as np
import matplotlib.pyplot as plt

def warp_lower_region(img, crop_ratio=0.8):
    h, w = img.shape[:2]

    y_start = int(h * (1 - crop_ratio))
    crop = img[y_start:h, :]

    hc, wc = crop.shape[:2]

    shrink = 0.3  #(0 - 0.3)

    src_pts = np.float32([
        [wc * shrink, 0],            # lewy góra (lekko zwężony)
        [wc * (1 - shrink), 0],     # prawy góra (lekko zwężony)
        [wc - 1, hc - 1],           # prawy dół
        [0, hc - 1]                # lewy dół  
    ])

    dst_pts = np.float32([
        [0, 0],
        [wc - 1, 0],
        [wc - 1, hc - 1],
        [0, hc - 1]
    ])

    H, _ = cv2.findHomography(src_pts, dst_pts)

    warped = cv2.warpPerspective(crop, H, (wc, hc))

    return crop, warped, src_pts, dst_pts

img = cv2.imread("from_ssh_machine/drone_visualization_results_seq22/overlays/000100_overlay.png")
if img is None:
    raise ValueError("I cannot read the image:(")

crop_ratio = 0.7
crop, warped, src_pts, dst_pts = warp_lower_region(img, crop_ratio)

plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
plt.title("Oryginał")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title(f"Dolne {crop_ratio} obrazu")
plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Po korekcji perspektywy")
plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
