from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import SGD
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary

from data_loader.DataLoader import DIV2K, GaoFen2, Sev2Mod, WV3, GaoFen2panformer
from PanNet import PanNet_model
from utils import *
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize DataLoader
    train_dataset = WV3(
        Path("/home/ubuntu/project/Data/WorldView3/train/train_wv3-001.h5"), transforms=[(RandomHorizontalFlip(1), 0.3), (RandomVerticalFlip(1), 0.3)]) #/home/ubuntu/project
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=16, shuffle=True, drop_last=True)

    validation_dataset = WV3(
        Path("/home/ubuntu/project/Data/WorldView3/val/valid_wv3.h5"))
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=16, shuffle=True)

    test_dataset = WV3(
        Path("/home/ubuntu/project/Data/WorldView3/drive-download-20230627T115841Z-001/test_wv3_multiExm1.h5"))
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False)

    # Initialize Model, optimizer, criterion and metrics
    model = PanNet_model(scale=4, ms_channels=8, mslr_mean=train_dataset.mslr_mean.to(device), mslr_std=train_dataset.mslr_std.to(device), pan_mean=train_dataset.pan_mean.to(device),
                     pan_std=train_dataset.pan_std.to(device)).to(device)


    optimizer = SGD(model.parameters(), lr=0.000007, momentum=0.9, weight_decay=10e-7)


    '''# initialize optimizers
            if args.optim_type == 'adam':
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=args.lr, betas=(args.beta, 0.999), weight_decay=args.weight_decay)
            elif args.optim_type == 'sgd':
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=args.lr, momentum=args.momentum)'''

    criterion = MSELoss().to(device)

    metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    val_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    test_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    tr_report_loss = 0
    val_report_loss = 0
    test_report_loss = 0
    tr_metrics = []
    val_metrics = []
    test_metrics = []
    best_eval_psnr = 0
    best_test_psnr = 0
    current_daytime = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    steps = 250000
    save_interval = 1000
    report_interval = 50
    test_intervals = [20000, 40000, 60000, 80000, 1000000, 120000, 140000, 160000, 180000, 2000000, 2200000, 2400000, 2500000]
    evaluation_interval = [20000, 40000, 60000, 80000, 1000000, 120000, 140000, 160000, 180000, 2000000, 2200000, 2400000, 2500000]
    
    lr_decay_intervals = [100000, 200000]
    val_steps = 50
    continue_from_checkpoint = True

    '''# Model summary
    pan_example = torch.randn(
        (1, 1, 256, 256)).to(device)
    mslr_example = torch.randn(
        (1, 4, 64, 64)).to(device)

    summary(model, pan_example, mslr_example, verbose=1)'''

    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    step = 250000

    # load checkpoint
    if continue_from_checkpoint:
        tr_metrics, val_metrics, test_metrics = load_checkpoint(torch.load(
            'checkpoints/pannet_model_WV3/pannet_model_WV3_2023_07_22-14_38_46.pth.tar'), model, optimizer, tr_metrics, val_metrics, test_metrics)
        print('Model Loaded ...')

    def scaleMinMax(x):
        return ((x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)))

    idx = 15 
    #WV3: 14
    # for gaofen: 15 or 8
    # evaluation mode
    model.eval()
    with torch.no_grad():
        test_iterator = iter(test_loader)
        for i, (pan, mslr, mshr) in enumerate(test_iterator):
            if idx == i:
                # forward
                pan, mslr, mshr = pan.to(device), mslr.to(
                    device), mshr.to(device)
                mssr = model(pan, mslr)
                test_loss = criterion(mssr, mshr)
                test_metric = test_metric_collection.forward(mssr, mshr)
                test_report_loss += test_loss

                # compute metrics
                test_metric = test_metric_collection.compute()

                figure, axis = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
                axis[0].imshow((scaleMinMax(mslr.permute(0, 3, 2, 1).detach().cpu()[
                                0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
                axis[0].set_title('(a) LR')
                axis[0].axis("off")

                axis[1].imshow(pan.permute(0, 3, 2, 1).detach().cpu()[
                                0, ...], cmap='gray')
                axis[1].set_title('(b) PAN')
                axis[1].axis("off")

                axis[2].imshow((scaleMinMax(mssr.permute(0, 3, 2, 1).detach().cpu()[
                                0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
                axis[2].set_title(
                    f'(c) PanNet {test_metric["psnr"]:.2f}dB/{test_metric["ssim"]:.4f}')
                axis[2].axis("off")

                axis[3].imshow((scaleMinMax(mshr.permute(0, 3, 2, 1).detach().cpu()[
                                0, ...].numpy())).astype(np.float32)[..., :3], cmap='viridis')
                axis[3].set_title('(d) GT')
                axis[3].axis("off")

                plt.savefig('results/Images_WV3.png')

                mslr = mslr.permute(0, 3, 2, 1).detach().cpu().numpy()
                pan = pan.permute(0, 3, 2, 1).detach().cpu().numpy()
                mssr = mssr.permute(0, 3, 2, 1).detach().cpu().numpy()
                gt = mshr.permute(0, 3, 2, 1).detach().cpu().numpy()

                np.savez('results/img_array_WV3.npz', mslr=mslr,
                         pan=pan, mssr=mssr, gt=gt)


if __name__ == '__main__':
    main()
