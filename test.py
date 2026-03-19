import argparse
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from networks.srn.net import SpeckleReductionNet
from utils.image_ops import linear_normalization


def load_model(model_path, device):
    model = SpeckleReductionNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def evaluate(model, noisy_images, clean_images, device):
    ssim_list = []
    psnr_list = []

    with torch.no_grad():
        for i in range(len(noisy_images)):
            norm_input = linear_normalization(noisy_images[i])
            tensor_input = torch.tensor(norm_input).unsqueeze(0).unsqueeze(0).to(device)

            output, _ = model(tensor_input, tensor_input)
            output = torch.clamp(output, 0, 1).squeeze().cpu().numpy()

            gt = linear_normalization(clean_images[i])

            ssim_val = ssim(gt, output, data_range=1.0)
            psnr_val = psnr(gt, output, data_range=1.0)

            ssim_list.append(ssim_val)
            psnr_list.append(psnr_val)

            print(f"[{i+1}/{len(noisy_images)}] SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB")

    return ssim_list, psnr_list


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    data = np.load(args.data_path)
    if data.ndim == 4 and data.shape[1] == 2:
        noisy_images = data[:, 0]
        clean_images = data[:, 1]
    else:
        raise ValueError("Test data must be paired with shape (N, 2, H, W)")

    print(f"Loaded {len(noisy_images)} test images from {args.data_path}")

    ssim_list, psnr_list = evaluate(model, noisy_images, clean_images, device)

    print(f"\n{'='*40}")
    print(f"Mean SSIM: {np.mean(ssim_list):.4f} (+/- {np.std(ssim_list):.4f})")
    print(f"Mean PSNR: {np.mean(psnr_list):.2f} dB (+/- {np.std(psnr_list):.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate denoising model with SSIM and PSNR.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test .npy file (N, 2, H, W)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model .pth file")
    args = parser.parse_args()

    main(args)
