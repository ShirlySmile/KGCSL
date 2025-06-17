import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module.contourlet import Contourlet
import numpy as np



def conv3x3_bn_relu(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.InstanceNorm2d(out_channel, affine=True),
        nn.ReLU(inplace=True),
    )


def conv5x5_bn_relu(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 5, 1, 2),
        nn.InstanceNorm2d(out_channel, affine=True),
        nn.ReLU(inplace=True),
    )

def conv7x7_bn_relu(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 7, 1, 3),
        nn.InstanceNorm2d(out_channel, affine=True),
        nn.ReLU(inplace=True),
    )


def bn_relu(out_channel):
    return nn.Sequential(
        nn.InstanceNorm2d(out_channel, affine=True),
        nn.ReLU(inplace=True),
    )


def conv1x1_bn_relu(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1),
        nn.InstanceNorm2d(out_channel, affine=True),
        nn.ReLU(inplace=True),
    )


def downsample2x(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1),
        nn.InstanceNorm2d(out_channel, affine=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )



class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=4, expansion_factor=1):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x




def repeat_block(block_channel, r=4):
    layers = [
        nn.Sequential(
            TransformerBlock(block_channel),
        )]
    return nn.Sequential(*layers)


class convfuse(nn.Module):
    def __init__(self, in_channels1, in_channels2, inner):
        super(convfuse, self).__init__()

        self.conv = conv1x1_bn_relu(in_channels1, inner)

    def forward(self, x, f):
        x1 = self.conv(x)
        out = x1 + f
        return out


class convpriorL(nn.Module):
    def __init__(self, in_channels, ou_channels):
        super(convpriorL, self).__init__()
        self.conv = conv5x5_bn_relu(in_channels, in_channels)
        self.conv1 = nn.Conv2d(in_channels, ou_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        x2 = self.conv1(x)
        return x2


class convprior(nn.Module):
    def __init__(self, in_channels, ou_channels):
        super(convprior, self).__init__()
        self.conv = conv1x1_bn_relu(in_channels, ou_channels)

    def forward(self, f, prior_h):
        x = self.conv(prior_h)
        out = f + x
        return out


class SCNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SCNet, self).__init__()

        self.Structureprior = Contourlet()

        block1_channels = 32
        block2_channels = 64
        block3_channels = 64
        block4_channels = 96


        self.conv1 = conv1x1_bn_relu(in_channels, block1_channels)
        self.conv2 = conv3x3_bn_relu(block1_channels, block1_channels)
        self.repeat_block1 = repeat_block(block1_channels)

        self.downsample1 = downsample2x(block1_channels, block2_channels)
        self.repeat_block2 = repeat_block(block2_channels)

        self.downsample2 = downsample2x(block2_channels, block3_channels)
        self.repeat_block3 = repeat_block(block3_channels)

        self.downsample3 = downsample2x(block3_channels, block4_channels)
        self.repeat_block4 = repeat_block(block4_channels)


        self.fuse_3x3convs = nn.ModuleList([
            convfuse(block4_channels, block3_channels, block3_channels),
            convfuse(block3_channels, block2_channels, block2_channels),
            convfuse(block2_channels, block1_channels, block1_channels),
        ])

        self.featureL_prior = nn.ModuleList([
            convpriorL(3, block2_channels),
            convpriorL(3, block3_channels),
            convpriorL(3, block4_channels),
        ])

        self.featureH_prior = nn.ModuleList([
            convprior(9, block4_channels),
            convprior(24, block3_channels),
            convprior(24, block2_channels),
        ])

        self.cls_pred_conv = nn.Conv2d(block1_channels, num_classes, 1)
        self.conv_bn_relu = TransformerBlock(block4_channels)


    def forward(self, x):
        FL, FH = self.Structureprior(x)

        feat_list = []
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.repeat_block1(x1)
        feat_list.append(x1)

        x2 = self.downsample1(x1)
        f1 = FL[2].to(x.device)
        x2 = x2 + self.featureL_prior[0](f1)
        x2 = self.repeat_block2(x2)
        feat_list.append(x2)

        x3 = self.downsample2(x2)
        f1 = FL[1].to(x.device)
        x3 = x3 + self.featureL_prior[1](f1)
        x3 = self.repeat_block3(x3)
        feat_list.append(x3)

        x4 = self.downsample3(x3)
        f1 = FL[0].to(x.device)
        x4 = x4 + self.featureL_prior[2](f1)
        x4 = self.repeat_block4(x4)
        feat_list.append(x4)

        inner_feat_list = feat_list
        inner_feat_list.reverse()

        feat = inner_feat_list[0]
        feat = self.conv_bn_relu(feat)
        out_feat_list = [feat]


        for i in range(len(inner_feat_list) - 1):
            b, c, h1, w1 = inner_feat_list[i + 1].size()
            b, c, h2, w2 = inner_feat_list[i].size()
            scale = h1 / h2
            fu = self.featureH_prior[i](out_feat_list[i], FH[i].to(x.device))
            inner0 = F.interpolate(fu, scale_factor=scale, mode='bilinear', align_corners=False)
            out = self.fuse_3x3convs[i](inner0, inner_feat_list[i + 1])
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]
        logit = self.cls_pred_conv(final_feat)
        return logit, final_feat


class SELCLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(SELCLoss, self).__init__()
        self.num_classes = num_classes
        self.soft_labels = None

        self.iter = 0
        self.momentum = 0.9
        self.eps = 1e-5
        self.initloss = nn.CrossEntropyLoss(ignore_index=-1).cuda()

    def calculate_entropy(self, probs):
        return -torch.sum(probs * torch.log(probs + self.eps), dim=1)



    def PrototypeContrastiveLoss(self, final_feat, labels):
        B, C, H, W = final_feat.shape
        final_feat = final_feat.view(C, -1)
        labels = labels.view(B, -1)

        prototypes = []
        for c in range(self.num_classes):
            mask = (labels == c)
            cc = mask.sum()
            if mask.sum() > 0:
                class_feat = final_feat[:, mask[0]]
                prototypes.append(class_feat.mean(dim=1))
        prototypes = torch.stack(prototypes).squeeze(1)

        distances = self.cosine_distances(final_feat.transpose(1, 0).contiguous(), prototypes)

        labels_flatten = labels.view(-1)
        unlabeled_indices = (labels_flatten == -1)
        unlabeled_indices = unlabeled_indices.unsqueeze(-1).expand_as(distances)

        unlabeled_distances = distances[unlabeled_indices].view(-1, self.num_classes)
        min_unlabeled_distances = unlabeled_distances.min(dim=1).values

        probabilities = torch.nn.functional.softmax(-unlabeled_distances, dim=1)

        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-7), dim=1)

        weights = torch.log(torch.tensor(self.num_classes).float()) - entropy

        weighted_unlabeled_loss = (min_unlabeled_distances * weights).mean()
        loss = weighted_unlabeled_loss
        return loss



    def forward(self, output, label):
        self.iter = self.iter + 1
        logits, final_feat = output
        loss_ce = self.initloss(logits, label)

        loss_prior = self.PrototypeContrastiveLoss(final_feat, label)

        loss_all = loss_ce + loss_prior*1000

        return loss_all


if __name__ == "__main__":

    HS = torch.randn(1, 103, 616, 344)

    y = torch.randint(0, 5, (1, 616, 344))
    grf_net = SCNet(in_channels=103, num_classes=9)
    grf_net.cuda()

    out = grf_net(HS.cuda())


