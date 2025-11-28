import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.writer.add_scalar(tag, value, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        self.writer.add_histogram(tag, values, step, bins=bins)

    def image_summary(self, tag, images, step):
        """Log a list of images: numpy or tensor, shape (B x C x H x W), [-1,1]"""
        self.writer.add_images(tag, images, step, dataformats="NCHW")

    def close(self):
        self.writer.close()
