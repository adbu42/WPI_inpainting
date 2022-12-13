import torch
from torch import nn
from torch import autograd
from models.discriminator import DiscriminatorDoubleColumn


# modified from WGAN-GP
def calc_gradient_penalty(netD, real_data, fake_data, masks, cuda, Lambda):
    batch_size = real_data.size()[0]
    dim = real_data.size()[2]
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, dim, dim)
    
    fake_data = fake_data.view(batch_size, 3, dim, dim)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates, masks)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    return gradient_penalty.sum().mean()


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


#tv loss
def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLossWithGAN(nn.Module):
    def __init__(self, extractor, Lamda, lr, betasInit=(0.5, 0.9)):
        super(InpaintingLossWithGAN, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.discriminator = DiscriminatorDoubleColumn(3)
        self.D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betasInit)
        self.cudaAvailable = torch.cuda.is_available()
        self.numOfGPUs = torch.cuda.device_count()
        self.lamda = Lamda

    def forward(self, input, mask, output, gt):
        self.discriminator.zero_grad()
        d_real = self.discriminator(gt, mask)
        d_real = d_real.mean().sum() * -1
        d_fake = self.discriminator(output, mask)
        d_fake = d_fake.mean().sum() * 1
        gp = calc_gradient_penalty(self.discriminator, gt, output, mask, self.cudaAvailable, self.lamda)
        d_loss = d_fake - d_real + gp
        self.D_optimizer.zero_grad()
        d_loss.backward(retain_graph=True)
        self.D_optimizer.step()
        
        output_comp = mask * input + (1 - mask) * output

        hole_loss = 6 * self.l1((1 - mask) * output, (1 - mask) * gt)
        valid_area_loss = self.l1(mask * output, mask * gt)

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray and RGB is possible')

        prc_loss = 0.0
        for i in range(3):
            prc_loss += 0.01 * self.l1(feat_output[i], feat_gt[i])
            prc_loss += 0.01 * self.l1(feat_output_comp[i], feat_gt[i])

        style_loss = 0.0
        for i in range(3):
            style_loss += 120 * self.l1(gram_matrix(feat_output[i]), gram_matrix(feat_gt[i]))
            style_loss += 120 * self.l1(gram_matrix(feat_output_comp[i]), gram_matrix(feat_gt[i]))

        g_loss = hole_loss + valid_area_loss + prc_loss + style_loss + 0.1 * d_fake
        return g_loss.sum(), d_loss.item(), hole_loss.item(), valid_area_loss.item(), prc_loss.item(), style_loss.item(), g_loss.item()