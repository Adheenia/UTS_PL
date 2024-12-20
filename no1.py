import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, img_as_ubyte
from skimage.color import rgb2gray
from google.colab import files

uploaded = files.upload()

if uploaded:
    file_name = list(uploaded.keys())[0]
    try:
        original_image = iio.imread(file_name)
        print(f"Image '{file_name}' loaded successfully!")
    except Exception as e:
        print(f"Error loading image: {e}")
else:
    print("No image uploaded.")

def custom_rgb2gray(image):
    if len(image.shape) == 3:
        return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    return image

gray_image = custom_rgb2gray(original_image)

def custom_histogram_equalization(image):
    image_flattened = image.flatten()
    hist, bins = np.histogram(image_flattened, bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    equalized = np.interp(image_flattened, bins[:-1], cdf_normalized * 255)
    return equalized.reshape(image.shape)

enhanced_image = custom_histogram_equalization(gray_image)

def adaptive_histogram_equalization(image):
    image_normalized = image / image.max()  
    image_byte = img_as_ubyte(image_normalized)
    return exposure.equalize_adapthist(image_byte, clip_limit=0.03)

clahe_image = adaptive_histogram_equalization(gray_image)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original Grayscale Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Enhanced Image (HE)")
plt.imshow(enhanced_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Adaptive Histogram Equalization (CLAHE)")
plt.imshow(clahe_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].hist(gray_image.flatten(), bins=256, range=[0, 256], color='gray')
axes[0].set_title('Histogram of Original Grayscale')

axes[1].hist(enhanced_image.flatten(), bins=256, range=[0, 256], color='gray')
axes[1].set_title('Histogram of Enhanced Image')

axes[2].hist(clahe_image.flatten(), bins=256, range=[0, 256], color='gray')
axes[2].set_title('Histogram of CLAHE Image')

plt.tight_layout()
plt.show()
