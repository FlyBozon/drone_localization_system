import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.cluster import KMeans

def match_histograms(source, reference):
    matched = np.zeros_like(source)

    for channel in range(3):
        matched[:, :, channel] = exposure.match_histograms(
            source[:, :, channel], 
            reference[:, :, channel]
        )

    return matched

reference_image = cv2.imread('datasets/UAVid/uavid_train/seq1/Images/000100.png')
ge_image = cv2.imread('output/dataset/gdansk_seq/Screenshot from 2025-11-24 19-35-49.png')

matched_img = match_histograms(ge_image, reference_image)

plt.subplot(3,3,1)
plt.imshow(reference_image)
plt.subplot(3,3,2)
plt.imshow(ge_image)
plt.subplot(3,3,3)
plt.imshow(matched_img)


def color_transfer(source, target):
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    source_mean, source_std = source_lab.mean(axis=(0,1)), source_lab.std(axis=(0,1))
    target_mean, target_std = target_lab.mean(axis=(0,1)), target_lab.std(axis=(0,1))

    result_lab = source_lab.copy()
    for i in range(3):
        result_lab[:,:,i] = ((source_lab[:,:,i] - source_mean[i]) * 
                            (target_std[i] / source_std[i])) + target_mean[i]

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    return result

reference = cv2.imread('datasets/UAVid/uavid_train/seq1/Images/000100.png')
ge_image = cv2.imread('output/dataset/gdansk_seq/Screenshot from 2025-11-24 19-35-49.png')
transferred = color_transfer(ge_image, reference)

plt.subplot(3,3,4)
plt.imshow(reference)
plt.subplot(3,3,5)
plt.imshow(ge_image)
plt.subplot(3,3,6)
plt.imshow(transferred)


class FastStyleTransfer:
    def __init__(self):
        print("6")
        self.model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    def transfer(self, content_image, style_image):
        print("1")
        content_image = tf.image.convert_image_dtype(content_image, tf.float32)
        style_image = tf.image.convert_image_dtype(style_image, tf.float32)

        print("2")

        content_image = content_image[tf.newaxis, :]
        style_image = style_image[tf.newaxis, :]
        print("3")

        stylized_image = self.model(content_image, style_image)[0]
        print("4")

        return stylized_image.numpy()

styler = FastStyleTransfer()

reference = cv2.imread('datasets/UAVid/uavid_train/seq1/Images/000100.png')
ge_image = cv2.imread('output/dataset/gdansk_seq/Screenshot from 2025-11-24 19-35-49.png')
print("5")
ge_image = ge_image[:, :, :3]
reference = reference[:, :, :3]
print("9")

# transferred1 = styler.transfer(ge_image, reference)
# transferred1 = transferred1[0]  

# plt.subplot(3,3,7)
# plt.imshow(reference)
# plt.subplot(3,3,8)
# plt.imshow(ge_image)
# plt.subplot(3,3,9)
# plt.imshow(transferred1)
# plt.show()

class SemanticHistogramMatcher:
    def __init__(self, reference_images):
        self.references = reference_images
        self.reference_features = self._extract_features()

    def _extract_features(self):
        features = []
        for ref in self.references:
            lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
            mean_color = lab.mean(axis=(0, 1))
            std_color = lab.std(axis=(0, 1))
            features.append(np.concatenate([mean_color, std_color]))
        return np.array(features)

    def _segment_by_color(self, image, n_clusters=5):
        pixels = image.reshape(-1, 3)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_pixels = lab.reshape(-1, 3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(lab_pixels)

        masks = []
        for i in range(n_clusters):
            mask = (labels == i).reshape(image.shape[:2])
            masks.append(mask)

        return masks, kmeans.cluster_centers_

    def match(self, source):
        masks, cluster_centers = self._segment_by_color(source, n_clusters=5)

        result = source.copy().astype(np.float32)

        for mask, center in zip(masks, cluster_centers):
            if mask.sum() == 0:
                continue

            best_ref = self._find_best_reference(center)

            for channel in range(3):
                source_region = source[:, :, channel][mask]
                ref_channel = best_ref[:, :, channel]

                if len(source_region) == 0:
                    continue

                matched_region = exposure.match_histograms(
                    source_region, 
                    ref_channel.flatten()
                )

                result[:, :, channel][mask] = matched_region

        return result.astype(np.uint8)

    def _find_best_reference(self, color_center):
        min_dist = float('inf')
        best_ref = self.references[0]

        for ref, features in zip(self.references, self.reference_features):
            dist = np.linalg.norm(features[:3] - color_center)

            if dist < min_dist:
                min_dist = dist
                best_ref = ref

        return best_ref

reference_images = []
for i in range(100, 120):  
    img = cv2.imread(f'datasets/UAVid/uavid_train/seq1/Images/{i:06d}.png')
    if img is not None:
        reference_images.append(img)


# matcher = SemanticHistogramMatcher(reference_images)
# ge_image = cv2.imread('output/dataset/gdansk_seq/Screenshot from 2025-11-24 19-35-49.png')
# matched = matcher.match(ge_image)

# # plt.figure(figsize=(15, 5))
# # plt.subplot(131)
# # plt.imshow(cv2.cvtColor(reference_images[0], cv2.COLOR_BGR2RGB))
# # plt.title('Reference')
# # plt.subplot(132)
# # plt.imshow(cv2.cvtColor(ge_image, cv2.COLOR_BGR2RGB))
# # plt.title('Google Earth')
# # plt.subplot(133)
# # plt.imshow(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
# # plt.title('Matched (Multi-Reference)')
# # plt.show()

# plt.subplot(3,3,7)
# plt.imshow(reference_images[0])
# plt.subplot(3,3,8)
# plt.imshow(ge_image)
# plt.subplot(3,3,9)
# plt.imshow(matched)
# plt.show()

class SemanticHistogramMatcher:
    def __init__(self, reference_images, n_clusters=6, alpha=0.4, method="moment"):

        self.references = reference_images
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.method = method
        self.reference_features = self._extract_features()

    def _extract_features(self):
        features = []
        for ref in self.references:
            lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
            mean_color = lab.mean(axis=(0, 1))
            std_color = lab.std(axis=(0, 1))
            features.append(np.concatenate([mean_color, std_color]))
        return np.array(features)

    def _segment_by_color(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        pixels = lab.reshape(-1, 3)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)

        masks = []
        for i in range(self.n_clusters):
            mask = (labels == i).reshape(image.shape[:2])
            masks.append(mask)

        return masks, kmeans.cluster_centers_

    def _find_best_reference(self, color_center):
        min_dist = float('inf')
        best_ref = self.references[0]

        for ref, feat in zip(self.references, self.reference_features):
            dist = np.linalg.norm(feat[:3] - color_center)
            if dist < min_dist:
                min_dist = dist
                best_ref = ref

        return best_ref

    def _moment_match(self, src, ref):
        src = src.astype(np.float32)
        ref = ref.astype(np.float32)

        src_m = src.mean()
        src_s = src.std() + 1e-5
        ref_m = ref.mean()
        ref_s = ref.std() + 1e-5

        out = (src - src_m) / src_s   
        out = out * ref_s + ref_m     
        return np.clip(out, 0, 255)

    def _histogram_match(self, src, ref):
        matched = exposure.match_histograms(src, ref, channel_axis=None)
        blended = self.alpha * matched + (1 - self.alpha) * src
        return np.clip(blended, 0, 255)

    def match(self, source):
        matches, centers = self._segment_by_color(source)
        result = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)

        for mask, center in zip(matches, centers):
            if mask.sum() == 0:
                continue

            best_ref = self._find_best_reference(center)
            best_ref_lab = cv2.cvtColor(best_ref, cv2.COLOR_BGR2LAB)

            for ch in range(3):
                src_region = result[:, :, ch][mask]
                ref_region = best_ref_lab[:, :, ch].flatten()

                if len(src_region) == 0:
                    continue

                if self.method == "moment":
                    matched = self._moment_match(src_region, ref_region)
                else:
                    matched = self._histogram_match(src_region, ref_region)

                blended = self.alpha * matched + (1 - self.alpha) * src_region
                result[:, :, ch][mask] = blended

        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


reference_images = []
for i in range(100, 120):
    img = cv2.imread(f'datasets/UAVid/uavid_train/seq1/Images/{i:06d}.png')
    if img is not None:
        reference_images.append(img)

matcher = SemanticHistogramMatcher(
    reference_images,
    n_clusters=7,      # 5–10
    alpha=0.35,        # intensity (0.2–0.5)
    method="moment"    #"histogram" #for classical matching  
)

ge_image = cv2.imread('output/dataset/gdansk_seq/Screenshot from 2025-11-24 19-35-49.png')
matched = matcher.match(ge_image)

plt.subplot(3,3,7)
plt.imshow(reference_images[0])
plt.subplot(3,3,8)
plt.imshow(ge_image)
plt.subplot(3,3,9)
plt.imshow(matched)
plt.show()
