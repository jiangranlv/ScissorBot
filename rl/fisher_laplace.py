import numpy as np
import torch
from os.path import join, basename, dirname, abspath
from rl.so3 import batch_torch_A_to_R
from torch.nn import functional as F

# We found SVD on cuda may have speed / stability (CUDA error encountered) issues, which is related to CUDA version, PyTorch version, etc., and decided to use SVD on cpu
CPUSVD = True
EPS = 1e-8

# grids_path = join(dirname(abspath(__file__)), 'grids3.npy')
# print(f'Loading SO3 discrete grids {grids_path}')
# GRIDS = torch.from_numpy(np.load(grids_path))
    
def fisher_nll(net_out, R, reduction):
    A = net_out.view(-1, 3, 3)
    loss_v = KL_Fisher(A, R, overreg=1.05)
    if reduction == 'mean':
        return torch.mean(loss_v)
    elif reduction == 'none':
        return loss_v

def laplace_nll(net_out, R, reduction):
    pred = net_out.reshape(-1, 3, 3)
    grids = GRIDS.to(R.device)
    logp = log_pdf('RLaplace', pred, R, grids)
    losses = -logp
    if reduction == 'mean':
        return torch.mean(losses)
    elif reduction == 'none':
        return losses
    
def vmf_loss(net_out, R, overreg=1.05):
    A = net_out.view(-1, 3, 3)
    loss_v = KL_Fisher(A, R, overreg=overreg)
    if loss_v is None:
        Rest = torch.unsqueeze(torch.eye(3, 3, device=R.device, dtype=R.dtype), 0)
        Rest = torch.repeat_interleave(Rest, R.shape[0], 0)
        return None, Rest

    Rest = batch_torch_A_to_R(A)
    return loss_v, Rest

_global_svd_fail_counter = 0
def KL_Fisher(A, R, overreg=1.05):
    # A is bx3x3
    # R is bx3x3
    global _global_svd_fail_counter
    try:
        A, R = A.cpu(), R.cpu()
        U, S, V = torch.svd(A)
        with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
            rotation_candidate = torch.matmul(U, V.transpose(1, 2))
            s3sign = torch.det(rotation_candidate)
        S_sign = S.clone()
        S_sign[:, 2] *= s3sign
        log_normalizer = logC_F(S_sign)
        log_exponent = -torch.matmul(A.view(-1, 1, 9), R.view(-1, 9, 1)).view(-1)
        _global_svd_fail_counter = max(0, _global_svd_fail_counter - 1)
        return (log_exponent + overreg * log_normalizer).cuda()
    except RuntimeError as e:
        _global_svd_fail_counter += 10  # we want to allow a few failures, but not consistent ones
        if _global_svd_fail_counter > 100:  # we seem to have gotten these problems more often than 10% of batches
            for i in range(A.shape[0]):
                print(A[i])
            raise e
        else:
            print('SVD returned NAN fail counter = {}'.format(_global_svd_fail_counter))
            return None
        

def delta_R(N):
    """
    volume for SO(3) is 1 (based on the normalization condition of Haar measure)
    volume for S^3 is 2*pi^2
    so delta_R is 1 / N, delta_q is 2*pi^2 / N
    """
    return 1 / N


def Laplace_NLL_loss(fn_type, pred, gt, grids):
    """
    @param pred: A from network (b, 3, 3)
    @param gt: gt matrices, (b, 3, 3)
    """
    pred = pred.reshape(-1, 3, 3)
    logp = log_pdf(fn_type, pred, gt, grids)
    losses = -logp

    pred_orth, _ = analytical_mode(pred, fn_type)

    return losses, pred_orth



def logF_const(power_fn, A, grids):
    # grids = grids.to(A.device)
    N = grids.shape[0]

    # change dimensionality for broadcasting
    grids1 = grids[None]  # (1, N, 3, 3)
    A1 = A[:, None]  # (b, 1, 3, 3)

    power = power_fn(A1, grids1)    # (b, N)
    # to avoid numerical explosion
    # logF = c + log(Sum{ exp(power-c) } * dR)
    c = power.max(dim=-1)[0]  # (b, )
    exps = torch.exp(power - c[:, None])    # (b, N)
    logF = c + torch.log(exps.sum(1) * delta_R(N))
    return logF


def logF_const_laplace(power_fn, A, grids):
    # grids = grids.to(A.device)
    N = grids.shape[0]

    # change dimensionality for broadcasting
    grids1 = grids[None]  # (1, N, 3, 3)
    A1 = A[:, None]  # (b, 1, 3, 3)

    power = power_fn(A1, grids1)    # (b, N)
    # to avoid numerical explosion
    # logF = c + log(Sum{ exp(power-c) } * dR)
    c = power.max(dim=-1)[0]  # (b, )
    exps = torch.exp(power - c[:, None])    # (b, N)
    logF = c + torch.log((exps / (-power)).sum(1) * delta_R(N))
    return logF



def log_pdf(fn_type, A, x, grids, broadcast=False):
    fn_dict = dict(
        RFisher=power_fn_fisher,
        RLaplace=power_fn_sqrtL2_proper,
    )
    power_fn = fn_dict[fn_type]

    if 'RLaplace' in fn_type:
        logF = logF_const_laplace(power_fn, A, grids)
    else:
        logF = logF_const(power_fn, A, grids)

    if x.shape[0] == grids.shape[0] or broadcast:
        # change dimensionality for broadcasting
        x = x[None]  # (1, N, 3, 3)
        A = A[:, None]  # (b, 1, 3, 3)
        logF = logF[:, None]    # (b, 1)
    if 'RLaplace' in fn_type:
        power = power_fn(A, x)
        pdf = -logF + power - torch.log(-power)
    else:
        pdf = -logF + power_fn(A, x)

    return pdf      # (b, ) if not broadcast; (b, N) if broadcast


def analytical_mode(pred, fn_type):
    device = pred.device
    if CPUSVD:
        pred = pred.cpu()

    U, S, VT = torch.linalg.svd(pred)
    with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(U, VT))
    diag = torch.stack((torch.ones_like(s3sign), torch.ones_like(s3sign), s3sign), -1)
    diag = torch.diag_embed(diag)

    pred_orth = U @ diag @ VT
    pred_orth = pred_orth.to(device)
    return pred_orth, s3sign


def power_fn_fisher(A, input):
    """
    power = tr(A^T x)
    To verify the discrete calculation. Should be consistent with fisher_utils.fisher_log_pdf()

    if x:
        @param A: (b, 3, 3)
        @param input: (b, 3, 3)
        @return logp: (b, )
    if grids:
        @param A: (b, 1, 3, 3)
        @param input: (1, N, 3, 3)
        @return logp: (b, N)
    """

    mul = torch.matmul(torch.transpose(A, -1, -2), input)
    assert mul.shape[-2:] == (3, 3)
    power = mul[..., 0, 0] + mul[..., 1, 1] + mul[..., 2, 2]

    return power


def power_fn_sqrtL2_proper(A, input):
    """
    power = -sqrt{ s1+s2+s3 - tr(A^T x) }

    if x:
        @param A: (b, 3, 3)
        @param input: (b, 3, 3)
        @return logp: (b, )
    if grids:
        @param A: (b, 1, 3, 3)
        @param input: (1, N, 3, 3)
        @return logp: (b, N)
    """
    mul = torch.matmul(torch.transpose(A, -1, -2), input)
    assert mul.shape[-2:] == (3, 3)
    tr = mul[..., 0, 0] + mul[..., 1, 1] + mul[..., 2, 2]

    device = A.device
    if CPUSVD:
        A = A.cpu()

    S = torch.linalg.svdvals(A)
    S = torch.cat((S[..., :-1], S[..., -1:] * torch.sign(torch.det(A))[..., None]), -1)

    s_sum = S.sum(-1)
    s_sum = s_sum.to(device)

    sqrt_min = (s_sum - tr).min()
    if not sqrt_min > -0.01:
        print(f'power_fn_sqrtL2: sqrt(negative numbers), min:{(s_sum - tr).min()}, num: {((s_sum - tr) < 0).sum()}')

    power = -torch.sqrt(torch.clamp_min(s_sum - tr, EPS))

    return power


def _horner(arr, x):
    z = torch.empty(x.shape, dtype=x.dtype, device=x.device).fill_(arr[0])
    for i in range(1, len(arr)):
        z.mul_(x).add_(arr[i])
    return z


torch_bessel0_a = [1.0, 3.5156229, 3.0899424, 1.2067492, 0.2659732, 0.360768e-1, 0.45813e-2][::-1]
torch_bessel0_b = [0.39894228, 0.1328592e-1, 0.225319e-2, -0.157565e-2, 0.916281e-2, -0.2057706e-1, 0.2635537e-1, -0.1647633e-1, 0.392377e-2][::-1]


def bessel0(x):  # always supressed by exp(x)
    # x is of size (-1)
    abs_x = torch.abs(x)
    mask = abs_x <= 3.75
    e1 = _horner(torch_bessel0_a, (abs_x / 3.75) ** 2) / torch.exp(abs_x)
    e2 = _horner(torch_bessel0_b, 3.75 / abs_x) / torch.sqrt(abs_x)
    e2[mask] = e1[mask]
    return e2


def torch_integral(f, v, from_x, to_x, N):
    with torch.no_grad():
        # computes ret_i = \int_{from_x}^{to_x} f(x,v_i)
        # where N is number of trapezoids + 1 per v_i
        rangee = torch.arange(N, dtype=v.dtype, device=v.device)
        x = (rangee * ((to_x - from_x) / (N - 1)) + from_x).view(1, N)
        weights = torch.empty((1, N), dtype=v.dtype, device=v.device).fill_(1)
        weights[0, 0] = 1 / 2
        weights[0, -1] = 1 / 2
        y = f(x, v)
        return torch.sum(y * weights, dim=1) * (to_x - from_x) / (N - 1)


def integrand_CF(x, s):
    # x is (1, N)
    # s is (-1, 3)
    # return (-1, N)
    # s is sorted from large to small
    f1 = (s[:, 1] - s[:, 2]) / 2
    f2 = (s[:, 1] + s[:, 2]) / 2
    a1 = f1.view(-1, 1) * (1 - x).view(1, -1)
    a2 = f2.view(-1, 1) * (1 + x).view(1, -1)
    a3 = (s[:, 2] + s[:, 0]).view(-1, 1) * (x - 1).view(1, -1)
    i1 = bessel0(a1)
    i2 = bessel0(a2)
    i3 = torch.exp(a3)
    ret = i1 * i2 * i3
    return ret


def integrand_Cdiff(x, s):
    s2 = s[:, 0]
    s1 = torch.max(s[:, 1:], dim=1).values
    s0 = torch.min(s[:, 1:], dim=1).values
    f1 = (s1 - s0) / 2
    f2 = (s1 + s0) / 2
    a1 = f1.view(-1, 1) * (1 - x).view(1, -1)
    a2 = f2.view(-1, 1) * (1 + x).view(1, -1)
    a3 = (s0 + s2).view(-1, 1) * (x - 1).view(1, -1)
    i1 = bessel0(a1)
    i2 = bessel0(a2)
    i3 = x.view(1, -1)
    i4 = torch.exp(a3)
    return i1 * i2 * i3 * i4


class class_logC_F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        N = 512
        # input is (..., 3) correspond to SINGULAR VALUES of F (NOT Lambda)
        shape = input.shape
        input_v = input.view(-1, 3)
        factor = 1 / 2 * torch_integral(integrand_CF, input_v, -1, 1, N)
        log_factor = torch.log(factor)
        log_supress = torch.sum(input_v, dim=1)
        ctx.save_for_backward(input, factor)
        return (log_factor + log_supress).view(shape[:-1])

    @staticmethod
    def backward(ctx, grad):
        S, factor = ctx.saved_tensors
        S = S.view(-1, 3)
        N = 512
        ret = torch.empty((S.shape[0], 3), dtype=S.dtype, device=S.device)
        for i in range(3):
            cv = torch.cat((S[:, i:], S[:, :i]), dim=1)
            ret[:, i] = 1 / 2 * torch_integral(integrand_Cdiff, cv, -1, 1, N)
        ret /= factor.view(-1, 1)
        ret *= grad.view(-1, 1)
        return ret.view((*grad.shape, 3))


logC_F = class_logC_F.apply