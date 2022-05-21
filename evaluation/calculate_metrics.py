from skimage.io import imread
from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import argparse

def calculateMetrics(reference, test):
    reference_img = imread(reference)
    test_img = imread(test)
    reference_img = img_as_float(reference_img)
    test_img = img_as_float(test_img)

    ssim_reference_test = ssim(reference_img, test_img, data_range=test_img.max()-test_img.min(), channel_axis=2)
    psnr_reference_test = psnr(reference_img, test_img, data_range=test_img.max()-test_img.min())
    print(f"SSIM: {ssim_reference_test}, PSNR: {psnr_reference_test}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', dest="reference", type=str)
    parser.add_argument('--test', dest="test", type=str)
    args = parser.parse_args()
    calculateMetrics(args.reference, args.test)