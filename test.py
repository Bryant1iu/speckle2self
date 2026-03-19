import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from networks.srn.net import SpeckleReductionNet
from utils.datasets import DenoisingDatasetPaired


def load_model(model_path, device):
    model = SpeckleReductionNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def evaluate(model, dataloader, device):
    ssim_list = []
    psnr_list = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            image_high = batch['image_high'].to(device)
            image_low = batch['image_low'].to(device)
            image_clean = batch['image_clean']

            output, _ = model(image_high, image_low)
            output = torch.clamp(output, -1, 1).squeeze(1).cpu().numpy()
            gt = image_clean.squeeze(1).numpy()

            for j in range(output.shape[0]):
                ssim_val = ssim(gt[j], output[j], data_range=2.0)
                psnr_val = psnr(gt[j], output[j], data_range=2.0)

                ssim_list.append(ssim_val)
                psnr_list.append(psnr_val)

                idx = i * dataloader.batch_size + j + 1
                print(f"[{idx}/{len(dataloader.dataset)}] SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB")

    return ssim_list, psnr_list


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)

    dataset = DenoisingDatasetPaired(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Loaded {len(dataset)} test images from {args.data_dir}")

    ssim_list, psnr_list = evaluate(model, dataloader, device)

    print(f"\n{'='*40}")
    print(f"Mean SSIM: {np.mean(ssim_list):.4f} (+/- {np.std(ssim_list):.4f})")
    print(f"Mean PSNR: {np.mean(psnr_list):.2f} dB (+/- {np.std(psnr_list):.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate denoising model with SSIM and PSNR.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to test directory (img_hr/, img_lr/, label/)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    args = parser.parse_args()

    main(args)
