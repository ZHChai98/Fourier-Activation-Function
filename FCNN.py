import torch
import torch.nn as nn

import numpy as np

'''
定义n阶傅里叶级数
'''
# n=30 ReLU
A = [-0.20264236728467555, 0.0, -0.02251581858718617, 0.0, -0.00810569469138702, 0.0, -0.0041355585160137865, 0.0,
     -0.0025017576207984637, 0.0, -0.0016747303081378145, 0.0, -0.0011990672620395006, 0.0, -0.0009006327434874469, 0.0,
     -0.0007011846618846905, 0.0, -0.0005613361974644753, 0.0, -0.0004595065017793097, 0.0, -0.00038306685687084223,
     0.0, -0.0003242277876554809, 0.0, -0.0002779730689776071, 0.0, -0.00024095406335871057, 0.0]
B = [0.3183098861837907, -0.15915494309189535, 0.1061032953945969, -0.07957747154594767, 0.06366197723675814,
     -0.05305164769729845, 0.04547284088339867, -0.039788735772973836, 0.0353677651315323, -0.03183098861837907,
     0.028937262380344612, -0.026525823848649224, 0.024485375860291588, -0.022736420441699334, 0.02122065907891938,
     -0.019894367886486918, 0.018724110951987685, -0.01768388256576615, 0.016753151904410037, -0.015915494309189534,
     0.015157613627799556, -0.014468631190172306, 0.013839560268860466, -0.013262911924324612, 0.012732395447351627,
     -0.012242687930145794, 0.0117892550438441, -0.011368210220849667, 0.010976202971854851, -0.01061032953945969]
w = [3.141592653589793]
a0 = [0.5]


# n=60 ReLU
# A = [-0.20264236728467555, 0.0, -0.02251581858718617, 0.0, -0.00810569469138702, 0.0, -0.0041355585160137865, 0.0,
#      -0.0025017576207984637, 0.0, -0.0016747303081378145, 0.0, -0.0011990672620395006, 0.0, -0.0009006327434874469, 0.0,
#      -0.0007011846618846905, 0.0, -0.0005613361974644753, 0.0, -0.0004595065017793097, 0.0, -0.00038306685687084223,
#      0.0, -0.0003242277876554809, 0.0, -0.0002779730689776071, 0.0, -0.00024095406335871057, 0.0,
#      -0.00021086614701839286, 0.0, -0.00018608114534864607, 0.0, -0.00016542234064055147, 0.0, -0.00014802218209253144,
#      0.0, -0.0001332296957821667, 0.0, -0.00012054870153758212, 0.0, -0.00010959565564341567, 0.0,
#      -0.00010007030483193855, 0.0, -9.173488786087621e-05, 0.0, -8.439915338803646e-05, 0.0, -7.790940687607672e-05,
#      0.0, -7.214039419176773e-05, 0.0, -6.698921232551258e-05, 0.0, -6.237068860716391e-05, 0.0,
#      -5.8213837197551146e-05, 0.0]
# B = [0.3183098861837907, -0.15915494309189535, 0.1061032953945969, -0.07957747154594767, 0.06366197723675814,
#      -0.05305164769729845, 0.04547284088339867, -0.039788735772973836, 0.0353677651315323, -0.03183098861837907,
#      0.028937262380344612, -0.026525823848649224, 0.024485375860291588, -0.022736420441699334, 0.02122065907891938,
#      -0.019894367886486918, 0.018724110951987685, -0.01768388256576615, 0.016753151904410037, -0.015915494309189534,
#      0.015157613627799556, -0.014468631190172306, 0.013839560268860466, -0.013262911924324612, 0.012732395447351627,
#      -0.012242687930145794, 0.0117892550438441, -0.011368210220849667, 0.010976202971854851, -0.01061032953945969,
#      0.01026806084463841, -0.009947183943243459, 0.009645754126781536, -0.009362055475993843, 0.009094568176679734,
#      -0.008841941282883075, 0.008602969896859207, -0.008376575952205019, 0.00816179195343053, -0.007957747154594767,
#      0.0077636557605802615, -0.007578806813899778, 0.007402555492646295, -0.007234315595086153, 0.007073553026306459,
#      -0.006919780134430233, 0.006772550769867886, -0.006631455962162306, 0.00649612012619981, -0.006366197723675813,
#      0.006241370317329229, -0.006121343965072897, 0.006005846909128126, -0.00589462752192205, 0.0057874524760689215,
#      -0.005684105110424833, 0.005584383968136679, -0.0054881014859274255, 0.005395082816674419, -0.005305164769729845]
# w = [3.141592653589793]
# a0 = [0.5]


# n=30 pre-trian
# A = [-1.4007e+01, -3.4718e+00, 6.5743e-01, 8.8805e-02, -3.0080e-01,
#      -3.5471e-01, -1.4764e-02, 2.6334e-01, 1.8092e-02, -3.2248e-03,
#      -2.0254e-01, -1.4549e-01, -2.5778e-01, 7.1596e-02, 4.7056e-01,
#      -2.2896e-01, -2.3136e-01, 3.5681e-01, 2.9199e-01, -3.1465e-01,
#      1.3936e-01, 4.0030e-01, 2.6467e-01, 1.4517e-01, -3.2120e-01,
#      -7.4065e-02, -2.1800e-01, 9.7215e-03, 3.6617e-01, -1.3458e-01]
#
# B = [1.0666e+02, 5.5523e-02, 1.5156e-01, 1.0732e+00, 1.3449e+00,
#      4.3107e-01, -1.6028e-01, 9.6653e-02, 9.3628e-02, -2.9941e-01,
#      -2.1513e-01, 1.6375e-01, 2.5644e-01, 5.6640e-01, 2.1762e-01,
#      -5.5805e-01, 3.3145e-01, 4.4965e-01, 6.2410e-02, 4.6896e-01,
#      1.1122e-01, -4.2424e-01, -2.3626e-01, 1.3257e-01, 1.6765e-01,
#      1.1034e-01, -3.3865e-01, 4.1961e-01, 3.5860e-02, -3.5997e-01]
#
# a0 = [-0.6680]
# w = [0.1082]


class Fourier(nn.Module):

    def __init__(self, N=30):
        super(Fourier, self).__init__()
        # 定义1*n的参数序列
        self.N = N
        self.n = torch.arange(start=1, end=N + 1, step=1, dtype=torch.float).cuda()

        # self.a0 = nn.Parameter(torch.randn(1), requires_grad=True)
        # self.w = nn.Parameter(torch.randn(1), requires_grad=True)
        # self.A = nn.Parameter(torch.randn(N), requires_grad=True)
        # self.B = nn.Parameter(torch.randn(N), requires_grad=True)

        self.a0 = nn.Parameter(torch.Tensor(a0[:N]), requires_grad=True)
        self.w = nn.Parameter(torch.Tensor(w[:N]), requires_grad=True)
        self.A = nn.Parameter(torch.Tensor(A[:N]), requires_grad=True)
        self.B = nn.Parameter(torch.Tensor(B[:N]), requires_grad=True)

        # self.a0 = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        # self.w = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        # self.A = nn.Parameter(torch.Tensor([1]*N), requires_grad=True)
        # self.B = nn.Parameter(torch.Tensor([1]*N), requires_grad=True)
        # print(self.w)
        # print(self.A)
        # print(self.B)
        # print(self.a0)

    def forward(self, input):
        # size变为N x 1 x 1 x 1 x 1
        n = self.n[..., None, None, None, None]
        A = self.A[..., None, None, None, None]
        B = self.B[..., None, None, None, None]

        # # size变为N x 1 x 1
        # n = self.n[..., None, None]
        # A = self.A[..., None, None]
        # B = self.B[..., None, None]

        nwt = n * self.w * input
        cosnwt = torch.cos(nwt)
        sinnwt = torch.sin(nwt)

        # output = A * cosnwt
        output = A * cosnwt + B * sinnwt

        output = output.sum(dim=0) + self.a0

        return output


class Sin(nn.Module):

    def __init__(self):
        super(Sin, self).__init__()

        self.C = nn.Parameter(torch.randn(1), requires_grad=True)
        self.w = nn.Parameter(torch.randn(1), requires_grad=True)
        self.A = nn.Parameter(torch.randn(1), requires_grad=True)
        self.B = nn.Parameter(torch.randn(1), requires_grad=True)

        # self.w = nn.Parameter(torch.Tensor(w[:1]), requires_grad=True)
        # self.A = nn.Parameter(torch.Tensor(A[:1]), requires_grad=True)
        # self.B = nn.Parameter(torch.Tensor(B[:1]), requires_grad=True)
        # self.C = nn.Parameter(torch.Tensor(a0[:1]), requires_grad=True)

        # self.a0 = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        # self.w = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        # self.A = nn.Parameter(torch.Tensor([1]*N), requires_grad=True)
        # self.B = nn.Parameter(torch.Tensor([1]*N), requires_grad=True)
        # print(self.w)
        # print(self.A)
        # print(self.B)
        # print(self.a0)

    def forward(self, input):
        # size变为N x 1 x 1 x 1 x 1
        A = self.A[..., None, None, None, None]
        B = self.B[..., None, None, None, None]

        # # size变为N x 1 x 1
        # n = self.n[..., None, None]
        # A = self.A[..., None, None]
        # B = self.B[..., None, None]

        nwt = self.w * input
        coswt = torch.cos(nwt)
        sinwt = torch.sin(nwt)

        # output = A * cosnwt
        output = A * torch.sin(B) * coswt + A * torch.cos(B) * sinwt

        output = output.sum(dim=0) + self.C

        return output


if __name__ == '__main__':
    f = Fourier(N=30).cuda()
    f = Sin().cuda()

    test = torch.randn(size=[2, 3, 3, 3]).cuda()
    # print(test)
    a1 = f.forward(test)
