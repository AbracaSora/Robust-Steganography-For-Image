import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from lpips import LPIPS
from kornia import color


# from taming.modules.losses.vqperceptual import *

class ImageSecretLoss(nn.Module):
    def __init__(self, recon_type='rgb', recon_weight=1., perceptual_weight=1.0, secret_weight=10., kl_weight=0.000001,
                 logvar_init=0.0, ramp=100000, max_image_weight_ratio=2.) -> None:
        super().__init__()
        self.recon_type = recon_type
        assert recon_type in ['rgb', 'yuv']
        if recon_type == 'yuv':
            self.register_buffer('yuv_scales', torch.tensor([1, 100, 100]).unsqueeze(1).float())  # [3,1]
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.secret_weight = secret_weight
        self.kl_weight = kl_weight

        self.ramp = ramp
        self.max_image_weight = max_image_weight_ratio * secret_weight - 1
        self.register_buffer('ramp_on', torch.tensor(False))
        self.register_buffer('step0', torch.tensor(1e9))  # large number

        self.perceptual_loss = LPIPS().eval()
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def activate_ramp(self, global_step):
        if not self.ramp_on:  # do not activate ramp twice
            self.step0 = torch.tensor(global_step)
            self.ramp_on = ~self.ramp_on
            print('[TRAINING] Activate ramp for image loss at step ', global_step)

    def compute_recon_loss(self, inputs, reconstructions):
        if self.recon_type == 'rgb':
            rec_loss = torch.abs(inputs - reconstructions).mean(dim=[1, 2, 3])
        elif self.recon_type == 'yuv':
            reconstructions_yuv = color.rgb_to_yuv((reconstructions + 1) / 2)
            inputs_yuv = color.rgb_to_yuv((inputs + 1) / 2)
            yuv_loss = torch.mean((reconstructions_yuv - inputs_yuv) ** 2, dim=[2, 3])
            rec_loss = torch.mm(yuv_loss, self.yuv_scales).squeeze(1)
        else:
            raise ValueError(f"Unknown recon type {self.recon_type}")
        return rec_loss

    def forward(self, inputs, reconstructions, posteriors, secret_gt, secret_pred, global_step):
        loss_dict = {}
        rec_loss = self.compute_recon_loss(inputs.contiguous(), reconstructions.contiguous())

        loss = rec_loss * self.recon_weight

        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous()).mean(dim=[1, 2, 3])
            loss += self.perceptual_weight * p_loss
            loss_dict['p_loss'] = p_loss.mean()

        loss = loss / torch.exp(self.logvar) + self.logvar
        if self.kl_weight > 0:
            kl_loss = posteriors.kl()
            loss += kl_loss * self.kl_weight
            loss_dict['kl_loss'] = kl_loss.mean()

        image_weight = 1 + min(self.max_image_weight,
                               max(0., self.max_image_weight * (global_step - self.step0.item()) / self.ramp))

        secret_loss = self.bce(secret_pred, secret_gt).mean(dim=1)
        loss = (loss * image_weight + secret_loss * self.secret_weight) / (image_weight + self.secret_weight)

        # loss dict update
        bit_acc = ((secret_pred.detach() > 0).float() == secret_gt).float().mean()
        loss_dict['bit_acc'] = bit_acc
        loss_dict['loss'] = loss.mean()
        loss_dict['img_lw'] = image_weight / self.secret_weight
        loss_dict['rec_loss'] = rec_loss.mean()
        loss_dict['secret_loss'] = secret_loss.mean()

        return loss.mean(), loss_dict


class ImageReconstructionLoss(torch.nn.Module):
    def __init__(self, recon_type='yuv', recon_weight=1.0, perceptual_weight=1.0, logvar_init=0.0, ramp=100000,
                 max_image_weight_ratio=2.0, secret_weight=2.0):
        super().__init__()
        assert recon_type in ['rgb', 'yuv']
        self.recon_type = recon_type
        if recon_type == 'yuv':
            self.register_buffer('yuv_scales', torch.tensor([1, 100, 100]).unsqueeze(1).float())  # 强化色彩通道
        self.recon_weight = recon_weight
        self.perceptual_weight = perceptual_weight
        self.secret_weight = secret_weight

        self.logvar = torch.nn.Parameter(torch.ones(size=()) * logvar_init)
        self.perceptual_loss = LPIPS().eval()
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()
        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=1.0).cpu()

        self.ramp = ramp
        self.max_image_weight = max_image_weight_ratio * secret_weight - 1
        self.register_buffer('ramp_on', torch.tensor(False))
        self.register_buffer('step0', torch.tensor(1e9))  # 不激活前不会变动

    def activate_ramp(self, global_step):
        if not self.ramp_on:
            self.step0 = torch.tensor(global_step)
            self.ramp_on = ~self.ramp_on
            print(f"[TRAINING] Ramp activated at step {global_step}")

    def compute_recon_loss(self, inputs, reconstructions):
        if self.recon_type == 'rgb':
            rec_loss = torch.abs(inputs - reconstructions).mean(dim=[1, 2, 3])
        elif self.recon_type == 'yuv':
            reconstructions_yuv = color.rgb_to_yuv((reconstructions + 1) / 2)
            inputs_yuv = color.rgb_to_yuv((inputs + 1) / 2)
            yuv_loss = torch.mean((reconstructions_yuv - inputs_yuv) ** 2, dim=[2, 3])
            rec_loss = torch.mm(yuv_loss, self.yuv_scales).squeeze(1)
        else:
            raise ValueError(f"Unknown recon type {self.recon_type}")
        return rec_loss

    def compute_secret_loss(self, secret, secret_recon):
        l1 = self.l1_loss(secret_recon, secret)
        ssim_loss = 1.0 - self.ssim(secret_recon, secret)
        l2 = self.mse_loss(secret_recon, secret)
        return l1, ssim_loss, l2
        # return l1, l2

    def forward(self, x, x_recon, global_step, secret=None, secret_recon=None):
        recon_loss = self.compute_recon_loss(x.contiguous(), x_recon.contiguous())
        # LPIPS 输入范围要求 [-1, 1]，确保一致
        x_input = x
        x_recon_input = x_recon
        # x_input = (x + 1) / 2
        # x_recon_input = (x_recon + 1) / 2

        loss = recon_loss * self.recon_weight

        perceptual = 0
        # perceptual = torch.clamp(self.perceptual_loss(x_recon_input.contiguous(), x_input.contiguous()), min=0).mean()
        if self.perceptual_weight > 0:
            perceptual = self.perceptual_loss(x_input.contiguous(), x_recon_input.contiguous()).mean(dim=[1, 2, 3])
            loss += self.perceptual_weight * perceptual

        # perceptual = self.perceptual_loss(x_recon, x).mean()

        # logvar loss scaling
        loss = loss / torch.exp(self.logvar) + self.logvar

        secret = secret.repeat(1,3,1,1)
        secret_recon = secret_recon.repeat(1,3,1,1)
        # # 设置白色阈值（0~1），一般 0.95 表示接近白色
        # white_threshold = 0.95
        #
        # # 生成权重 mask：非白色区域权重大（白色像素通常值为1）
        # # 如果图像是 RGB，这里取通道均值作为亮度近似（也可以取最大通道）
        # brightness = secret.mean(dim=1, keepdim=True)  # shape: [B, 1, H, W]
        # mask = (brightness < white_threshold).float()  # 非白区域为1，其余为0
        #
        # # 权重定义：白区域权重为1，非白区域权重为1+extra_weight
        # extra_weight = 10.0
        # weight = 1.0 + extra_weight * mask  # shape: [B, 1, H, W]
        #
        # # 计算加权 MSE 损失
        # mse = (secret_recon - secret) ** 2
        # secret_loss = (mse * weight).mean()

        secret_loss = 0.0
        l1 = 0.0
        ssim = 0.0
        l2 = 0.0
        if secret is not None and secret_recon is not None:
            l1, ssim, l2 = self.compute_secret_loss(secret.contiguous(), secret_recon.contiguous())
            secret_loss = (l1 * 0.2 + l2 * 0.2 + ssim + self.perceptual_loss(secret_recon.contiguous(), secret.contiguous()).mean(dim=[1, 2, 3]))
        # # dynamic ramp weight
        # if global_step >= self.step0.item():
        #     weight = 1 + min(self.max_weight, self.max_weight * (global_step - self.step0.item()) / self.ramp)
        # else:
        #     weight = 1.0  # 初始阶段未激活 ramp

        image_weight = 1 + min(self.max_image_weight,
                               max(0., self.max_image_weight * (global_step - self.step0.item()) / self.ramp))

        total_loss = (loss.mean() * image_weight + secret_loss * self.secret_weight) / (
                image_weight + self.secret_weight)

        # if perceptual.item() < 0:
        #     print(f"perceptual:{perceptual}, perceptual_loss(x_recon, x):{self.perceptual_loss(x_recon.contiguous(), x.contiguous())}")
        #     diff = torch.abs(x.contiguous() - x_recon.contiguous()).mean()
        #     print("Avg abs diff between x and x_recon:", diff.item())

        return total_loss, {
            'recon_loss': recon_loss.mean(),
            'perceptual_loss': perceptual.mean(),
            'l1': l1.item() if isinstance(l1, torch.Tensor) else 0.0,
            'ssim': ssim.item() if isinstance(ssim, torch.Tensor) else 0.0,
            'l2': l2.item() if isinstance(l2, torch.Tensor) else 0.0,
            'final_loss': total_loss.item(),
            # 'logvar': self.logvar.item(),
            'image_weight': image_weight,
            'image_loss': loss.mean(),
            'secret_loss': secret_loss
        }
