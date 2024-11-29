import torch
from torch import nn
import math
import torch.nn.functional as F

def conv1d(x, weight,stride, pad):
    n, c, h_in = x.shape
    d, c, k = weight.shape
    x_pad = torch.zeros(n, c, h_in+2*pad)
    if pad>0:
        x_pad[:, :, pad:-pad] = x
    else:
        x_pad = x

    x_pad = x_pad.unfold(2, k, stride)
    out = torch.einsum(
        'nchwkj,dckj->ndhw',
        x_pad, weight)
    out = out
    return out

def conv2d(x, weight, bias, stride, pad):
    n, c, h_in, w_in = x.shape
    d, c, k, j = weight.shape
    x_pad = torch.zeros(n, c, h_in+2*pad, w_in+2*pad)
    if pad>0:
        x_pad[:, :, pad:-pad, pad:-pad] = x
    else:
        x_pad = x

    x_pad = x_pad.unfold(2, k, stride)
    x_pad = x_pad.unfold(3, j, stride)
    out = torch.einsum(
        'nchwkj,dckj->ndhw',
        x_pad, weight)
    out = out + bias.view(1, -1, 1, 1)
    return out

class ASConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(ASConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv1 = nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride) #Get scaling factor A
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)
        self.modulation = modulation

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    # Generates relative coordinates around the center point of the convolution: pn
    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # Convolution center coordinates: p0
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = p_0_x.view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = p_0_y.view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, A, dtype): #amplified based on scaling factor A(floating point number), P_n
        N, h, w = offset.shape[1] // 2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + A*p_n
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.shape[3]
        c = x.shape[1]
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N]
        index = index * padded_w
        index = index + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        x_offset = x.gather(dim=-1, index=index)
        x_offset=x_offset.contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)
        return x_offset

    def forward(self, x):
        x = x
        offset = self.p_conv(x)
        A = self.p_conv1(x)
        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.shape[1] // 2
        if self.padding:
            x = self.zero_padding(x)
        # p = self._get_p(offset, dtype)
        x = x
        p = self._get_p(offset, A, dtype)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        a=x.size(2)
        b=x.size(3)
        A=q_lt[..., :N]
        B=q_lt[..., N:]
        # Clamps all elements in input into the range [ min, max ]
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                          torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))
        # Gets the coordinates of the 4 pixels closest to the offset pixel
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        # The pixel value after calculating the bilinear difference
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)
        return out

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv1d(inp_dim, out_dim,  k, padding=pad, stride=stride,
                              bias=not with_bn)
        self.bn = nn.BatchNorm1d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(batch_first=True, input_size=17, hidden_size=30, num_layers=1)
        self.attenion = Attention3dBlock()
        self.linear = nn.Sequential(
            nn.Linear(in_features=900, out_features=30),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=30, out_features=10),
            nn.ReLU(inplace=True)
        )
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=17, out_features=1),
            nn.ReLU(inplace=True)
        )
        self.handcrafted = nn.Sequential(
            nn.Linear(in_features=34, out_features=10),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),

        )
        self.output = nn.Sequential(
            nn.Linear(in_features=20, out_features=1)
        )

    def forward(self, inputs, handcrafted_feature):
        hf = self.handcrafted(handcrafted_feature)
        x = inputs
        x = x.reshape(-1, 30, 17, 1)
        y = x.permute(0, 2, 1, 3)
        ASC = ASConv2d(inc=30, outc=30, kernel_size=3)
        x1 = ASC(x)
        x = x1.reshape(-1, 510)
        x1 = x1.reshape(-1, 30, 17)
        conv =convolution(1, 30, 1, with_bn=True)
        A1 =conv(x1)
        ASC = ASConv2d(inc=17, outc=17, kernel_size=3)
        y1 = ASC(y)
        y = y1.reshape(-1, 510)
        y1 = y1.reshape(-1, 17, 30)
        conv = convolution(1, 17, 1, with_bn=True)
        A2 = conv(y1)
        A = torch.cat([A1, A2], 2)
        att = F.softmax(A, dim=2)
        attA1=att[...,:17]
        attA2 = att[..., 17:]
        X1 = torch.mul(x1,attA1)
        X2 = torch.mul(y1,attA2)
        X2 = X2.permute(0, 2, 1)
        f1= X1+X2
        f, (hn, cn) = self.lstm(f1)
        f = f.reshape(-1,900)
        x = self.linear(f)
        out = torch.cat((x, hf), dim=1)
        out_pre = self.output(out)
        conv1 = convolution(3, 30, 1, with_bn=True)#k,in,out
        K_R = conv1(f1)
        K_R = K_R.reshape(-1, 17)
        K_R = self.linear1(K_R)#100,1
        Tanh=nn.Tanh()
        r = Tanh(K_R)
        r = (r*3/4-1/4)*0.1
        out=r*out_pre
        return out

class Attention3dBlock(nn.Module):
    def __init__(self):
        super(Attention3dBlock, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_features=30, out_features=30),
            nn.Softmax(dim=2),
        )

    # inputs: batch size * window size(time step) * lstm output dims
    def forward(self, inputs):
        x = inputs.permute(0, 2, 1)
        x = self.linear(x)
        x_probs = x.permute(0, 2, 1)
        # print(torch.sum(x_probs.item()))
        output = x_probs * inputs
        return output