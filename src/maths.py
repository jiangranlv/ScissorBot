import taichi as ti
import numpy as np
import time
import math
import os
from typing import Tuple

import trimesh
import trimesh.transformations as tra

from src.sparse import *
from src.utils import *


@ti.dataclass
class tiTensor333:
    """3x3x3 tensor. data stored as one 9x3 matrix"""
    data: ti.types.matrix(9, 3, ti.f32)

    @ti.func
    def get(self, k, i, j):
        return self.data[i * 3 + j, k]

    @ti.func
    def set(self, k, i, j, v):
        self.data[i * 3 + j, k] = v


@ti.dataclass
class tiTensor3333:
    """3x3x3x3 tensor. data stored as three 3x9 matrices"""
    data0: ti.types.matrix(3, 9, ti.f32)
    data1: ti.types.matrix(3, 9, ti.f32)
    data2: ti.types.matrix(3, 9, ti.f32)

    @ti.func
    def get(self, k, l, i, j):
        ret_val = 0.0
        if i == 0:
            ret_val = self.data0[j, k * 3 + l]
        elif i == 1:
            ret_val = self.data1[j, k * 3 + l]
        elif i == 2:
            ret_val = self.data2[j, k * 3 + l]
        else:
            assert False, "i={} out of range".format(i)
        return ret_val

    @ti.func
    def set(self, k, l, i, j, v):
        if i == 0:
            self.data0[j, k * 3 + l] = v
        elif i == 1:
            self.data1[j, k * 3 + l] = v
        elif i == 2:
            self.data2[j, k * 3 + l] = v
        else:
            assert False, "i={} out of range".format(i)


@ti.dataclass
class tiMatrix9x9:
    """9x9 matrix. data stored as three 3x9 matrices"""
    data0: ti.types.matrix(3, 9, ti.f32)
    data1: ti.types.matrix(3, 9, ti.f32)
    data2: ti.types.matrix(3, 9, ti.f32)

    @ti.func
    def get(self, i, j):
        assert 0 <= i and i < 9 and 0 <= j and j < 9
        ret_val = 0.0
        i1 = i // 3
        i2 = i - i1 * 3
        if i1 == 0:
            ret_val = self.data0[i2, j]
        elif i1 == 1:
            ret_val = self.data1[i2, j]
        elif i1 == 2:
            ret_val = self.data2[i2, j]
        else:
            assert False, "[ERROR] tiMatrix9x9 error"
        return ret_val

    @ti.func
    def set(self, i, j, v):
        assert 0 <= i and i < 9 and 0 <= j and j < 9
        i1 = i // 3
        i2 = i - i1 * 3
        if i1 == 0:
            self.data0[i2, j] = v
        elif i1 == 1:
            self.data1[i2, j] = v
        elif i1 == 2:
            self.data2[i2, j] = v
        else:
            assert False, "[ERROR] tiMatrix9x9 error"


@ti.func
def ssvd_func(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V


@ti.func
def dsvd_func(F: ti.types.matrix(3, 3, ti.f32), tol: ti.f32):
    """
    Calculate svd of F and derivative of U, S, V.
        - F = U @ S @ VT
        - dU[k,l,i,j] = partial Ukl / partial Fij
        - dS[k,i,j] = partial Sk / partial Fij
        - dV[k,l,i,j] = partial Vkl / partial Fij

    Args:
        - F: 3x3 ti.Matrix
        - tol: positive numerical error tolerance

    Return:
        - U: 3x3 ti.Matrix
        - S: 3x3 ti.Matrix
        - V: 3x3 ti.Matrix
        - dU: tiTensor3333
        - dS: tiTensor333
        - dV: tiTensor3333
    """
    U, S, V = ssvd_func(F)
    dU = tiTensor3333(0.0)
    dS = tiTensor333(0.0)
    dV = tiTensor3333(0.0)

    if ti.abs(S[0, 0] - S[1, 1]) < tol or ti.abs(S[1, 1] - S[2, 2]) < tol or \
            ti.abs(S[2, 2] - S[0, 0]) < tol:
        for i in range(3):
            for j in range(3):
                F[i, j] += ti.random() * tol
        U, S, V = ssvd_func(F)

    w01 = 0.0
    w02 = 0.0
    w12 = 0.0

    d01 = S[1, 1] * S[1, 1] - S[0, 0] * S[0, 0]
    d02 = S[2, 2] * S[2, 2] - S[0, 0] * S[0, 0]
    d12 = S[2, 2] * S[2, 2] - S[1, 1] * S[1, 1]

    if ti.abs(d01) < tol:
        d01 = 0.0
    else:
        d01 = 1.0 / d01

    if ti.abs(d02) < tol:
        d02 = 0.0
    else:
        d02 = 1.0 / d02

    if ti.abs(d12) < tol:
        d12 = 0.0
    else:
        d12 = 1.0 / d12

    for r in range(3):
        for s in range(3):
            Ur = ti.Vector([U[r, 0], U[r, 1], U[r, 2]])
            Vs = ti.Vector([V[s, 0], V[s, 1], V[s, 2]])
            UVT = Ur.outer_product(Vs)

            # Compute dS
            for i in range(3):
                dS.set(i, r, s, UVT[i, i])

            for i in range(3):
                UVT[i, i] -= dS.get(r, s, i)

            tmp = S @ UVT + UVT.transpose() @ S
            w01 = tmp[0, 1] * d01
            w02 = tmp[0, 2] * d02
            w12 = tmp[1, 2] * d12

            tmp = ti.Matrix(
                [[0.0, w01, w02], [-w01, 0.0, w12], [-w02, -w12, 0.0]])
            V_tmp = V @ tmp
            for i in range(3):
                for j in range(3):
                    dV.set(i, j, r, s, V_tmp[i, j])

            tmp = UVT @ S + S @ UVT.transpose()
            w01 = tmp[0, 1] * d01
            w02 = tmp[0, 2] * d02
            w12 = tmp[1, 2] * d12

            tmp = ti.Matrix(
                [[0.0, w01, w02], [-w01, 0.0, w12], [-w02, -w12, 0.0]])
            U_tmp = U @ tmp
            for i in range(3):
                for j in range(3):
                    dU.set(i, j, r, s, U_tmp[i, j])
    return U, S, V, dU, dS, dV


@ti.func
def ldlt_decompose_9x9_func(A_mat: tiMatrix9x9, tol: ti.f32):
    """
    LDLT decompose for 9x9 matrix.

    Args:
        - mat: tiMatrix9x9
        - tol: positive numerical error tolerance

    Return:
        - is_success: bool
        - L_mat: tiMatrix9x9
        - D_vec: 9D ti.Vector
    """
    M = ti.static(9)

    L_mat = tiMatrix9x9()
    D_vec = ti.Vector.zero(ti.f32, M)
    is_success = True
    for j in ti.static(range(M)):
        # calculate Dj
        tmp_sum = 0.0
        for k in ti.static(range(j)):
            tmp_sum += L_mat.get(j, k) ** 2 * D_vec[k]
        D_vec[j] = A_mat.get(j, j) - tmp_sum

        # calculate Lij
        L_mat.set(j, j, 1.0)
        if D_vec[j] <= tol:
            D_vec[j] = 0.0
            for i in ti.static(range(j + 1, M)):
                L_mat.set(i, j, 0.0)
            is_success = False

        else:
            for i in ti.static(range(j + 1, M)):
                tmp_sum = 0.0
                for k in ti.static(range(j)):
                    tmp_sum += L_mat.get(i, k) * L_mat.get(j, k) * D_vec[k]
                lij = (A_mat.get(i, j) - tmp_sum) / D_vec[j]
                L_mat.set(i, j, lij)

    return is_success, L_mat, D_vec


@ti.func
def safe_normalized(a, eps) -> ti.Vector:
    """
    Args:
        - a: ti.Vector 3D
        - eps: float

    Return:
        - a / ti.max(a.norm(), eps)"""
    return a / ti.max(a.norm(), eps)


@ti.func
def get_2D_barycentric_weights_func(x, a, b, dx_eps) -> ti.Vector:
    """
    Args:
        - x, a, b: ti.Vector
        - dx_eps: ti.f32

    Return:
        - 2D ti.Vector (u, v)
            - u + v = 1.0
            - u * a + v * b = proj(x)
    """
    e = b - a
    t = e.dot(x - a) / ti.max(e.norm_sqr(), dx_eps ** 2)
    return ti.Vector([1.0 - t, t])


@ti.func
def get_3D_barycentric_weights_func(x, a, b, c, dx_eps) -> ti.Vector:
    """
    Args:
        - x, a, b, c: ti.Vector
        - dx_eps: ti.f32

    Return:
        - 3D ti.Vector (u, v, w)
            - u + v + w = 1.0
            - u * a + v * b + w * c = proj(x)
    """
    n = safe_normalized((b - a).cross(c - a), dx_eps)

    u = (b - x).cross(c - x).dot(n)
    v = (c - x).cross(a - x).dot(n)
    w = (a - x).cross(b - x).dot(n)

    uvw = u + v + w
    if uvw >= 0.0 and uvw < dx_eps:
        uvw = dx_eps
    elif uvw < 0.0 and uvw > - dx_eps:
        uvw = -dx_eps
    return ti.Vector([u / uvw, v / uvw, 1.0 - (u / uvw + v / uvw)])


@ti.func
def get_distance_func(x, a, b, is_segment, dx_eps) -> ti.f32:
    """
    Args:
        - x, a, b: ti.Vector
        - dx_eps: ti.f32

    Return:
        - ti.f32
    """
    bc = get_2D_barycentric_weights_func(x, a, b, dx_eps)
    if is_segment:
        bc = ti.math.clamp(bc, 0.0, 1.0)
    xp = bc[0] * a + bc[1] * b
    return (x - xp).norm()


@ti.func
def get_distance_vec_func(x, a, b, is_segment, dx_eps) -> ti.Vector:
    """
    Args:
        - x, a, b: ti.Vector
        - dx_eps: ti.f32

    Return:
        - ti.Vector, point from x to edge (proj(x) - x)
    """
    bc = get_2D_barycentric_weights_func(x, a, b, dx_eps)
    if is_segment:
        bc = ti.math.clamp(bc, 0.0, 1.0)
    xp = bc[0] * a + bc[1] * b
    return xp - x


@ti.func
def get_distance_to_triangle_func(x, a, b, c, dx_eps) -> ti.Vector:
    """
    Args:
        - x, a, b, c: ti.Vector
        - dx_eps: ti.f32

    Return: (l, u, v, w)
        - l = min(||x-d||), where d is in triangle abc
        - (u, v, w): barycentric coordinate of d
    """
    dist_a = ti.math.length(x - a)
    dist_b = ti.math.length(x - b)
    dist_c = ti.math.length(x - c)

    dist_ab = get_distance_func(x, a, b, True, dx_eps)
    dist_bc = get_distance_func(x, b, c, True, dx_eps)
    dist_ca = get_distance_func(x, c, a, True, dx_eps)

    dist = ti.min(dist_a, dist_b, dist_c, dist_ab, dist_bc, dist_ca)
    ret_bc = ti.Vector.zero(ti.f32, 3)
    if dist == dist_a:
        ret_bc = ti.Vector([1.0, 0.0, 0.0], ti.f32)
    elif dist == dist_b:
        ret_bc = ti.Vector([0.0, 1.0, 0.0], ti.f32)
    elif dist == dist_c:
        ret_bc = ti.Vector([0.0, 0.0, 1.0], ti.f32)
    elif dist == dist_ab:
        uu, vv = ti.math.clamp(get_2D_barycentric_weights_func(
            x, a, b, dx_eps), 0.0, 1.0)
        ret_bc = ti.Vector([uu, vv, 0.0], ti.f32)
    elif dist == dist_bc:
        vv, ww = ti.math.clamp(get_2D_barycentric_weights_func(
            x, b, c, dx_eps), 0.0, 1.0)
        ret_bc = ti.Vector([0.0, vv, ww], ti.f32)
    elif dist == dist_ca:
        ww, uu = ti.math.clamp(get_2D_barycentric_weights_func(
            x, c, a, dx_eps), 0.0, 1.0)
        ret_bc = ti.Vector([uu, 0.0, ww], ti.f32)
    u, v, w = get_3D_barycentric_weights_func(x, a, b, c, dx_eps)

    proj = u * a + v * b + w * c
    dist_p = ti.math.length(proj - x)

    if dist_p < dist and \
            0.0 < u and u < 1.0 and \
            0.0 < v and v < 1.0 and \
            0.0 < w and w < 1.0:
        dist = dist_p
        ret_bc = ti.Vector([u, v, w], ti.f32)

    return ti.Vector([dist, ret_bc[0], ret_bc[1], ret_bc[2]], ti.f32)


@ti.func
def get_intersect_func(x0, x1, y0, y1, dx_eps):
    """
    Args:
        - x0, x1, y0, y1: three-dimensional ti.Vector

    Return:
        - bc: float
            - intersection point is (bc * y0 + (1 - bc) * y1)
    """
    n = safe_normalized((x1 - x0).cross(y1 - y0), dx_eps)
    hx = safe_normalized((x1 - x0).cross(n), dx_eps)
    ha_vec = get_distance_vec_func(y0, x0, x1, False, dx_eps)
    hb_vec = get_distance_vec_func(y1, x0, x1, False, dx_eps)
    ha = ha_vec.dot(hx)
    hb = hb_vec.dot(-hx)

    ret_val = 0.5

    if ti.abs(ha + hb) > dx_eps:
        ret_val = hb / (ha + hb)

    return ret_val


@ti.func
def get_scaling_matrix_func(vec, scale, dx_eps):
    """
    Get scaling matrix. ||vec||_2 may not necessarily be 1.0
        - n = normalized(vec), a = scale
        - S = I + (a - 1.0) * n.outer_product(n)

    Args:
        - vec: 3D vector
        - scale: ti.f32

    Return:
        - mat: 3x3 matrix
    """
    ret_val = ti.Matrix.zero(ti.f32, 3, 3)
    ret_val[0, 0] = ret_val[1, 1] = ret_val[2, 2] = 1.0
    vec_n = safe_normalized(vec, dx_eps)
    ret_val += (scale - 1.0) * vec_n.outer_product(vec_n)
    return ret_val


def theta_phi_to_direc(theta, phi) -> np.ndarray:
    return np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])


def direc_to_theta_phi(direc: np.ndarray):
    x, y, z = direc
    theta = math.atan2(math.sqrt(x ** 2 + y ** 2), z)
    phi = math.atan2(y, x)
    return theta, phi

def compute_relative_rotation(current, target):
    current = current / np.linalg.norm(current)
    target = target / np.linalg.norm(target)

    axis = np.cross(current, target)
    axis = axis / np.linalg.norm(axis)
    
    # if np.dot(current, target) > 1 or np.dot(current, target) < -1:
    #     print('Error in computing relative rotation, current:{} target:{}'.format(current, target))
    #     return None
    dot_product = np.clip(np.dot(current, target), -1, 1)
    
    angle = np.arccos(dot_product)
    # print(current, axis, angle)
    theta, phi = direc_to_theta_phi(axis)
    
    return angle, theta, phi


def axis_angle_to_matrix(angle, theta, phi) -> np.ndarray:
    return tra.rotation_matrix(angle, theta_phi_to_direc(theta, phi))


@ti.func
def vec_add_vec_func(ans, l, a, k, b, n):
    """ans = l * a + k * b

    ans, a, b are vectors."""
    for i in range(n):
        ans[i] = l * a[i] + k * b[i]


@ti.kernel
def vec_add_vec_kernel(ans: ti.template(), l: ti.f32, a: ti.template(), k: ti.f32, b: ti.template(), n: ti.i32):
    """ans = l * a + k * b

    ans, a, b are vectors."""
    vec_add_vec_func(ans, l, a, k, b, n)


@ti.func
def vec_add_vec_batch_func(batch_size, batch_mask, ans, l, a, k, b, n_field):
    """ans = l * a + k * b

    ans, a, b are vectors."""
    max_n = vec_max_func(n_field, batch_size)
    for _b, i in ti.ndrange(batch_size, max_n):
        if i < n_field[_b] and batch_mask[_b] != 0.0:
            ans[_b, i] = l[_b] * a[_b, i] + k[_b] * b[_b, i]


@ti.kernel
def vec_add_vec_batch_kernel(batch_size: ti.i32, batch_mask: ti.template(), ans: ti.template(), l: ti.template(), a: ti.template(), k: ti.template(), b: ti.template(), n_field: ti.template()):
    """ans = l * a + k * b

    ans, a, b are vectors."""
    vec_add_vec_batch_func(batch_size, batch_mask, ans, l, a, k, b, n_field)


@ti.func
def vec_dot_vec_func(a, b, n):
    """ans = aT @ b"""
    ans = 0.0
    for i in range(n):
        ans += a[i] * b[i]
    return ans


@ti.kernel
def vec_dot_vec_kernel(a: ti.template(), b: ti.template(), n: ti.i32) -> ti.f32:
    """ans = aT @ b"""
    return vec_dot_vec_func(a, b, n)


@ti.func
def vec_dot_vec_batch_func(batch_size, batch_mask, a, b, ans, n_field):
    """ans = aT @ b"""
    for _b in range(batch_size):
        if batch_mask[_b] != 0.0:
            ans[_b] = 0.0

    max_n = vec_max_func(n_field, batch_size)
    for _b, i in ti.ndrange(batch_size, max_n):
        if i < n_field[_b] and batch_mask[_b] != 0.0:
            ans[_b] += a[_b, i] * b[_b, i]


@ti.kernel
def vec_dot_vec_batch_kernel(batch_size: ti.i32, batch_mask: ti.template(), a: ti.template(), b: ti.template(), ans: ti.template(), n_field: ti.template()):
    """ans = aT @ b"""
    return vec_dot_vec_batch_func(batch_size, batch_mask, a, b, ans, n_field)


@ti.func
def vec_dot_vec_kahan_func(a, b, n):
    """ans = aT @ b using kahan summation"""
    '''ans = 0.0
    y = 0.0
    ti.loop_config(serialize=True)
    for i in range(n):
        y -= a[i] * b[i]
        r = ans - y
        y = (r - ans) + y
        ans = r
    return ans'''
    ans = 0.0
    res = 0.0
    for i in range(n):
        mul = a[i] * b[i]
        old_sum = ti.atomic_add(ans, mul)
        if ti.abs(old_sum) > ti.abs(mul):
            res += ((old_sum + mul) - old_sum) - mul
        else:
            res += ((old_sum + mul) - mul) - old_sum
    return ans - res


@ti.kernel
def vec_dot_vec_kahan_kernel(a: ti.template(), b: ti.template(), n: ti.i32) -> ti.f32:
    """ans = aT @ b using kahan summation"""
    return vec_dot_vec_kahan_func(a, b, n)


@ti.func
def vec_dot_vec_kahan_batch_func(batch_size, batch_mask, a, b, ans, res, n_field):
    """ans = aT @ b using kahan summation"""
    '''ans = 0.0
    y = 0.0
    ti.loop_config(serialize=True)
    for i in range(n):
        y -= a[i] * b[i]
        r = ans - y
        y = (r - ans) + y
        ans = r
    return ans'''
    for _b in range(batch_size):
        if batch_mask[_b] != 0.0:
            ans[_b] = 0.0
            res[_b] = 0.0

    max_n = vec_max_func(n_field, batch_size)
    for _b, i in ti.ndrange(batch_size, max_n):
        if i < n_field[_b] and batch_mask[_b] != 0.0:
            mul = a[_b, i] * b[_b, i]
            old_sum = ti.atomic_add(ans[_b], mul)
            if ti.abs(old_sum) > ti.abs(mul):
                res[_b] += ((old_sum + mul) - old_sum) - mul
            else:
                res[_b] += ((old_sum + mul) - mul) - old_sum

    for _b in range(batch_size):
        if batch_mask[_b] != 0.0:
            ans[_b] -= res[_b]


@ti.kernel
def vec_dot_vec_kahan_batch_kernel(batch_size: ti.i32, batch_mask: ti.template(), a: ti.template(), b: ti.template(), ans: ti.template(), res: ti.template(), n_field: ti.template()):
    """ans = aT @ b using kahan summation"""
    return vec_dot_vec_kahan_batch_func(batch_size, batch_mask, a, b, ans, res, n_field)


@ti.func
def vec_div_vec_func(ans, a, b, n, tol):
    """ans = a / b (element wise)"""
    for i in range(n):
        if ti.abs(b[i]) >= ti.abs(tol):
            ans[i] = a[i] / b[i]
        else:
            ans[i] = ti.math.sign(b[i]) * a[i] / ti.abs(tol)
        '''elif b[i] > 0.0:
            ans[i] = a[i] / ti.abs(tol)
        else:
            ans[i] = -a[i] / ti.abs(tol)'''


@ti.kernel
def vec_div_vec_kernel(ans: ti.template(), a: ti.template(), b: ti.template(), n: ti.i32, tol: ti.f32):
    """ans = a / b (element wise)"""
    vec_div_vec_func(ans, a, b, n, tol)


@ti.func
def vec_mul_vec_func(ans, a, b, n):
    """ans = a * b (element wise)"""
    for i in range(n):
        ans[i] = a[i] * b[i]


@ti.kernel
def vec_mul_vec_kernel(ans: ti.template(), a: ti.template(), b: ti.template(), n: ti.i32):
    """ans = a * b (element wise)"""
    vec_mul_vec_func(ans, a, b, n)


@ti.func
def vec_mul_vec_batch_func(batch_size, batch_mask, ans, a, b, n_field):
    """ans = a * b (element wise)"""
    max_n = vec_max_func(n_field, batch_size)
    for _b, i in ti.ndrange(batch_size, max_n):
        if i < n_field[_b] and batch_mask[_b] != 0.0:
            ans[_b, i] = a[_b, i] * b[_b, i]


@ti.kernel
def vec_mul_vec_batch_kernel(batch_size: ti.i32, batch_mask: ti.template(), ans: ti.template(), a: ti.template(), b: ti.template(), n_field: ti.template()):
    """ans = a * b (element wise)"""
    vec_mul_vec_batch_func(batch_size, batch_mask, ans, a, b, n_field)


@ti.func
def block_mul_vec_func(ans, block, vec, n, block_dim):
    """
    M = block_dim

    ans[i * M + j] += block[i, j, k] * vec[j * M + k]
    """
    for i in range(n):
        ans[i] = 0.0
    num_block = (n + block_dim - 1) // block_dim
    for i, j, k in ti.ndrange(num_block, block_dim, block_dim):
        jj = i * block_dim + j
        kk = i * block_dim + k
        if jj < n and kk < n:
            ans[jj] += block[i, j, k] * vec[kk]


@ti.kernel
def block_mul_vec_kernel(ans: ti.template(), block: ti.template(), vec: ti.template(), n: ti.i32, block_dim: ti.i32):
    """
    M = block_dim

    ans[i * M + j] += block[i, j, k] * vec[j * M + k]
    """
    block_mul_vec_func(ans, block, vec, n, block_dim)


@ti.func
def block_mul_vec_batch_func(batch_size, batch_mask, ans, block, vec, n_field, block_dim):
    """
    M = block_dim

    ans[b, i * M + j] += block[b, i, j, k] * vec[b, j * M + k]
    """
    max_n = vec_max_func(n_field, batch_size)
    for _b, i in ti.ndrange(batch_size, max_n):
        if i < n_field[_b] and batch_mask[_b] != 0.0:
            ans[_b, i] = 0.0

    num_block = (max_n + block_dim - 1) // block_dim
    for _b, i, j, k in ti.ndrange(batch_size, num_block, block_dim, block_dim):
        if batch_mask[_b] != 0.0:
            jj = i * block_dim + j
            kk = i * block_dim + k
            if jj < n_field[_b] and kk < n_field[_b]:
                ans[_b, jj] += block[_b, i, j, k] * vec[_b, kk]


@ti.kernel
def block_mul_vec_batch_kernel(batch_size: ti.i32, batch_mask: ti.template(), ans: ti.template(), block: ti.template(), vec: ti.template(), n_field: ti.template(), block_dim: ti.i32):
    """
    M = block_dim

    ans[b, i * M + j] += block[b, i, j, k] * vec[b, j * M + k]
    """
    block_mul_vec_batch_func(batch_size, batch_mask, ans, block, vec, n_field, block_dim)


@ti.func
def vec_copy_func(dst, src, n):
    """dst = src"""
    for i in range(n):
        dst[i] = src[i]


@ti.kernel
def vec_copy_kernel(dst: ti.template(), src: ti.template(), n: ti.i32):
    """dst = src"""
    vec_copy_func(dst, src, n)


@ti.func
def vec_copy_batch_func(batch_size, dst, src, n_field):
    """dst = src"""
    max_n = vec_max_func(n_field, batch_size)
    for _b, i in ti.ndrange(batch_size, max_n):
        if i < n_field[_b]:
            dst[_b, i] = src[_b, i]


@ti.kernel
def vec_copy_batch_kernel(batch_size: ti.i32, dst: ti.template(), src: ti.template(), n_field: ti.template()):
    """dst = src"""
    vec_copy_batch_func(batch_size, dst, src, n_field)


@ti.func
def num_div_vec_func(ans, l, vec, n, tol):
    """ans = l / vec"""
    for i in range(n):
        if ti.abs(vec[i]) >= ti.abs(tol):
            ans[i] = l / vec[i]
        else:
            ans[i] = ti.math.sign(vec[i]) * l / ti.abs(tol)
        '''elif vec[i] > 0.0:
            ans[i] = l / ti.abs(tol)
        else:
            ans[i] = -l / ti.abs(tol)'''


@ti.kernel
def num_div_vec_kernel(ans: ti.template(), l: ti.f32, vec: ti.template(), n: ti.i32, tol: ti.f32):
    """ans = l / vec"""
    num_div_vec_func(ans, l, vec, n, tol)


@ti.func
def num_div_vec_batch_func(batch_size, ans, l, vec, n_field, tol):
    """ans = l / vec"""
    max_n = vec_max_func(n_field, batch_size)
    for _b, i in ti.ndrange(batch_size, max_n):
        if i < n_field[_b]:
            if ti.abs(vec[i]) >= ti.abs(tol):
                ans[_b, i] = l[_b] / vec[_b, i]
            else:
                ans[_b, i] = ti.math.sign(vec[_b, i]) * l[_b] / ti.abs(tol)


@ti.kernel
def num_div_vec_batch_kernel(batch_size: ti.i32, ans: ti.template(), l: ti.template(), vec: ti.template(), n_field: ti.template(), tol: ti.f32):
    """ans = l / vec"""
    num_div_vec_batch_func(batch_size, ans, l, vec, n_field, tol)


@ti.func
def num_mul_vec_func(ans, l, vec, n):
    """ans = l * vec"""
    for i in range(n):
        ans[i] = l * vec[i]


@ti.kernel
def num_mul_vec_kernel(ans: ti.template(), l: ti.f32, vec: ti.template(), n: ti.i32):
    """ans = l * vec"""
    num_mul_vec_func(ans, l, vec, n)


@ti.func
def vec_max_func(f: ti.template(), n: ti.i32) -> ti.f32:
    ret_val = f[0]
    for i in range(1, n):
        _ = ti.atomic_max(ret_val, f[i])
    return ret_val


@ti.kernel
def vec_max_kernel(f: ti.template(), n: ti.i32) -> ti.f32:
    return vec_max_func(f, n)


@ti.func
def vec_min_func(f: ti.template(), n: ti.i32) -> ti.f32:
    ret_val = ti.math.inf
    for i in range(n):
        _ = ti.atomic_min(ret_val, f[i])
    return ret_val


@ti.kernel
def vec_min_kernel(f: ti.template(), n: ti.i32) -> ti.f32:
    return vec_min_func(f, n)


@ti.func
def vec_abs_less_than_vec_all_func(a: ti.template(), b: ti.template(), n: ti.i32) -> bool:
    """return if abs(a[i]) < abs(b[i]) for all i in range(n)"""
    ret_val = True
    for i in range(n):
        if not (ti.abs(a[i]) < ti.abs(b[i])):
            ret_val = False
    return ret_val


@ti.func
def vec_sum_func(f: ti.template(), s: ti.i32, e: ti.i32) -> ti.f32:
    ret_val = 0.0
    for i in range(s, e):
        ret_val += f[i]
    return ret_val


@ti.kernel
def vec_sum_kernel(f: ti.template(), s: ti.i32, e: ti.i32) -> ti.f32:
    return vec_sum_func(f, s, e)


@ti.func
def vec_sum_batch_same_len_func(batch_size: ti.i32, f: ti.template(), ans: ti.template(), s: ti.i32, e: ti.i32):
    for _b in range(batch_size):
        ans[_b] = 0.0
    for _b, i in ti.ndrange(batch_size, (s, e)):
        ans[_b] += f[_b, i]


@ti.kernel
def vec_abs_less_than_vec_all_kernel(a: ti.template(), b: ti.template(), n: ti.i32) -> bool:
    """return if abs(a[i]) < abs(b[i]) for all i in range(n)"""
    return vec_abs_less_than_vec_all_func(a, b, n)


@ti.func
def in_unit_cube_func(coor: ti.types.vector(3, ti.f32)):
    """return if coor is in a unit cube, [0,1) x [0,1) x [0,1)"""
    ret_val = True
    for i in range(3):
        ret_val = ret_val and 0.0 <= coor[i] and coor[i] < 1.0
    return ret_val


@ti.kernel
def in_unit_cube_kernel(coor: ti.types.vector(3, ti.f32)) -> bool:
    """return if coor is in a unit cube, [0,1) x [0,1) x [0,1)"""
    return in_unit_cube_func(coor)


def trilinear(coor: np.ndarray, val: np.ndarray) -> np.float32:
    """trilinear interpolation.

    Args:
        - coor: 3x1 vector
        - val: 8x1 vector"""
    c = np.zeros((2, 2), np.float32)

    for j in range(2):
        for k in range(2):
            tmp = 2 * j + k
            c[j, k] = val[tmp] * (1 - coor[0]) + val[4 + tmp] * coor[0]

    d = np.zeros((2, ), np.float32)
    for k in range(2):
        d[k] = c[0, k] * (1 - coor[1]) + c[1, k] * coor[1]

    return d[0] * (1 - coor[2]) + d[1] * coor[2]


@ti.func
def trilinear_func(coor: ti.types.vector(3, ti.f32), val: ti.types.vector(8, ti.f32)) -> ti.f32:
    """trilinear interpolation.

    Args:
        - coor: 3x1 vector
        - val: 8x1 vector"""
    c = ti.Matrix(arr=[[0.0, 0.0], [0.0, 0.0]], dt=ti.f32)
    for j in range(2):
        for k in range(2):
            tmp = 2 * j + k
            c[j, k] = val[tmp] * (1 - coor[0]) + val[4 + tmp] * coor[0]

    d = ti.Vector(arr=[0.0, 0.0], dt=ti.f32)
    for k in range(2):
        d[k] = c[0, k] * (1 - coor[1]) + c[1, k] * coor[1]

    return d[0] * (1 - coor[2]) + d[1] * coor[2]


@ti.kernel
def trilinear_kernel(coor: ti.types.vector(3, ti.f32), val: ti.types.vector(8, ti.f32)) -> ti.f32:
    """trilinear interpolation.

    Args:
        - coor: 3x1 vector
        - val: 8x1 vector"""
    return trilinear_func(coor, val)


@ti.func
def trilinear_4D_func(coor: ti.types.vector(3, ti.f32), val: ti.types.matrix(8, 4, ti.f32)) -> ti.types.vector(4, ti.f32):
    """trilinear interpolation.

    Args:
        - coor: 3x1 vector
        - val: 8x4 vector

    Return:
        4D vector"""
    c = ti.Matrix.zero(dt=ti.f32, n=4, m=4)
    for j in range(2):
        for k in range(2):
            tmp = 2 * j + k
            c[tmp, :] = val[tmp, :] * (1 - coor[0]) + val[4 + tmp, :] * coor[0]

    d = ti.Matrix.zero(dt=ti.f32, n=2, m=4)
    for k in range(2):
        d[k, :] = c[k, :] * (1 - coor[1]) + c[2 + k, :] * coor[1]

    return d[0, :] * (1 - coor[2]) + d[1, :] * coor[2]


@ti.kernel
def trilinear_4D_kernel(coor: ti.types.vector(3, ti.f32), val: ti.types.matrix(8, 4, ti.f32)) -> ti.types.vector(4, ti.f32):
    """trilinear interpolation.

    Args:
        - coor: 3x1 vector
        - val: 8x4 vector

    Return:
        4D vector"""
    return trilinear_4D_func(coor, val)

def smooth_interp(a: float, b: float, n: int) -> np.ndarray:
    """
    Similar to `np.linspace(a, b, n)`, but use a smoother function (cubic) such that:

    `f[0]=a, f[n-1]=b, f'[0]=f'[n-1]=0`

    In contrast, if `f=np.linspace(a, b, n)`, then `f'[0]=f'[n-1]=(b-a)/n`
    """
    t = np.linspace(0, 1, n, dtype=float)
    p = 3 * t ** 2 - 2 * t ** 3
    return p * b + (1 - p) * a


def point_to_homo(pos: np.ndarray) -> np.ndarray:
    """3x1 -> 4x1 vector, pad 1.0"""
    return np.array([pos[0], pos[1], pos[2], 1.0])


@ti.func
def point_to_homo_func(pos: ti.types.vector(3, ti.f32)) -> ti.types.vector(4, ti.f32):
    """3x1 -> 4x1 vector, pad 1.0"""
    return ti.Vector([pos[0], pos[1], pos[2], 1.0], dt=ti.f32)


@ti.kernel
def point_to_homo_kernel(pos: ti.types.vector(3, ti.f32)) -> ti.types.vector(4, ti.f32):
    """3x1 -> 4x1 vector, pad 1.0"""
    return point_to_homo_func(pos)


def vector_to_homo(pos: np.ndarray) -> np.ndarray:
    """3x1 -> 4x1 vector, pad 0.0"""
    return np.array([pos[0], pos[1], pos[2], 0.0])


@ti.func
def vector_to_homo_func(pos: ti.types.vector(3, ti.f32)) -> ti.types.vector(4, ti.f32):
    """3x1 -> 4x1 vector, pad 0.0"""
    return ti.Vector([pos[0], pos[1], pos[2], 0.0], dt=ti.f32)


@ti.kernel
def vector_to_homo_kernel(pos: ti.types.vector(3, ti.f32)) -> ti.types.vector(4, ti.f32):
    """3x1 -> 4x1 vector, pad 0.0"""
    return point_to_homo_func(pos)

@ti.func
def upper_bound_func(x, f, sizen):
    """
    Return the position of first element `y` in `f` such that `y > x`
    Args:
        x: target number
        f: sorted list
        sizen: size of f

    Example:
        >>> x = 10
        >>> a = ti.field([1, 5, 8, 9, 30, 100])
        >>>                           ^
        >>>                         x = 10
        >>> return 4
        >>> 
        >>> x = 10
        >>> a = ti.field([1, 5, 8, 10, 10, 100])
        >>>                                 ^
        >>>                               x = 10
        >>> return 3
    """
    left = 0
    right = sizen - 1
    ti.loop_config(serialize=True)
    while (left <= right):
        mid = (left + right) // 2
        if f[mid] > x:
            right = mid - 1
        else:
            left = mid + 1
    return left

@ti.kernel
def upper_bound_kernel(x: ti.template(), f: ti.template(), sizen: ti.i32) -> ti.i32:
    """
    Return the position of first element `y` in `f` such that `y > x`
    Args:
        x: target number
        f: sorted list
        sizen: size of f

    Example:
        >>> x = 10
        >>> a = ti.field([1, 5, 8, 9, 30, 100])
        >>>                           ^
        >>>                         x = 10
        >>> return 4
        >>> 
        >>> x = 10
        >>> a = ti.field([1, 5, 8, 10, 10, 100])
        >>>                                 ^
        >>>                               x = 10
        >>> return 3
    """
    return upper_bound_func(x, f, sizen)

@ti.func
def get_accumulate_func(x, x_acc, n):
    x_acc[0] = x[0]
    ti.loop_config(serialize=True)
    for i in range(1, n):
        x_acc[i] = x_acc[i - 1] + x[i]

@ti.func
def get_accumulate_masked_func(x_mask, x, x_acc, n):
    x_acc[0] = x[0]
    ti.loop_config(serialize=True)
    for i in range(1, n):
        if x_mask[i] != 0.0:
            x_acc[i] = x_acc[i - 1] + x[i]
        else:
            x_acc[i] = x_acc[i - 1]

@ti.func
def get_batch_and_idx(ib, batch_size, n_accumulate):
    """
    ib = n_accumulate[b - 1] + i

    return b, i
    """
    b = upper_bound_func(ib, n_accumulate, batch_size)
    i = ib
    if b > 0:
        i -= n_accumulate[b - 1]
    return b, i

@ti.func
def barrier_function_func(x: ti.f32, d: ti.f32, k: ti.f32) -> ti.types.vector(3, ti.f32):
    ret_val = ti.Vector.zero(dt=ti.f32, n=3)

    assert k >= 2
    if x < d:
        ret_val[0] = ((d - x) / d) ** k * d ** 2
        ret_val[1] = - k * ((d - x) / d) ** (k - 1) * d
        ret_val[2] = k * (k - 1) * ((d - x) / d) ** (k - 2)

    return ret_val


@ti.kernel
def barrier_function_kernel(x: ti.f32, d: ti.f32, k: ti.f32) -> ti.types.vector(3, ti.f32):
    return barrier_function_func(x, d, k)
