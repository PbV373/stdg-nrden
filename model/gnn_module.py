import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv_gwnet(nn.Module):
    def __init__(self):
        super(nconv_gwnet, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class nconv_gcnm_dynamic(nn.Module):
    def __init__(self):
        super(nconv_gcnm_dynamic, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nvwl->ncwl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn_gcnm_dynamic(nn.Module):
    def __init__(self, c_in=32, c_out=32, dropout=0.3, support_len=3, order=2):
        super(gcn_gcnm_dynamic, self).__init__()
        self.nconv_dynamic = nconv_gcnm_dynamic()
        self.nconv_static = nconv_gwnet()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support[:2]:  # Dynamic Adjacency Matrix
            x1 = self.nconv_dynamic(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv_dynamic(x1, a)
                out.append(x2)
                x1 = x2
        if len(support) > 2:
            for a in support[2:]:  # Pre-defined static adjacency matrix
                x1 = self.nconv_static(x, a)
                out.append(x1)
                for k in range(2, self.order + 1):
                    x2 = self.nconv_static(x1, a)
                    out.append(x2)
                    x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class gcn_gwnet(nn.Module):
    def __init__(self, c_in=32, c_out=32, dropout=0.3, support_len=3, order=2):
        super(gcn_gwnet, self).__init__()
        self.nconv = nconv_gwnet()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h