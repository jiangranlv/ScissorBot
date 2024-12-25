import taichi as ti
import numpy as np
import ctypes
import os
import shutil

from src.sparse import *
from src.utils import *
from src.maths import *

JACOBI_PRECOND = 1
INCOMPLETE_CHOLESKY_PRECOND = 2
BLOCK_JACOBI_PRECOND = 3


@ti.data_oriented
class Modified_PCG_sparse_solver:
    def __init__(self, batch_size: int, nmax: int, nmax_triplet: int, precond: int, dx_eps: float, dA_eps: float, max_iter: int, judge_plateau_avg_num: int, print_warning: bool) -> None:
        """Modified_PCG solver. solve Ax=b.
        A: n x n semi-positive definite matrix
        b: n
        """
        self.batch_size = batch_size
        self.print_warning = print_warning
        self.clock = Clock()
        
        assert nmax >= 1
        self.nmax = nmax
        self.n_field = ti.field(ti.i32, shape=(self.batch_size, ))
        self.n_field.fill(0)
        self.n_max_field = ti.field(ti.i32, shape=())
        self.n_max_field.fill(0)

        assert precond in [JACOBI_PRECOND,
                           INCOMPLETE_CHOLESKY_PRECOND, BLOCK_JACOBI_PRECOND], NotImplemented
        self.precond = precond

        self.A = SparseMatrix(self.batch_size, nmax, nmax, nmax_triplet, True)
        self.A.set_zero_kernel()
        self.b = ti.field(dtype=ti.f32, shape=(self.batch_size, nmax))
        self.b.fill(0.0)
        self.eps_f = ti.field(dtype=ti.f32, shape=(self.batch_size, ))
        self.eps_f.fill(0.0)
        self.con_f = ti.field(dtype=ti.f32, shape=(self.batch_size, nmax))
        self.con_f.fill(0.0)
        if precond == JACOBI_PRECOND:
            self.p_f = ti.field(dtype=ti.f32, shape=(self.batch_size, nmax))  # jacobi precond
            self.p_f.fill(0.0)
        elif precond == BLOCK_JACOBI_PRECOND:
            self.block_dim_jacobi = 3
            self.num_block_jacobi = (nmax + 2) // 3
            self.p_block_f = ti.field(dtype=ti.f32, shape=(
                self.batch_size, self.num_block_jacobi, 3, 3))
            """[B, BLOCK_NUM, 3, 3]"""
            self.p_block_f.fill(0.0)
        elif precond == INCOMPLETE_CHOLESKY_PRECOND:
            self.cdlls = None
        else:
            raise NotImplementedError

        self.x_f = ti.field(dtype=ti.f32, shape=(self.batch_size, nmax))
        self.x_best_f = ti.field(dtype=ti.f32, shape=(self.batch_size, nmax))
        self.r_f = ti.field(dtype=ti.f32, shape=(self.batch_size, nmax))
        self.c_f = ti.field(dtype=ti.f32, shape=(self.batch_size, nmax))
        self.q_f = ti.field(dtype=ti.f32, shape=(self.batch_size, nmax))
        self.s_f = ti.field(dtype=ti.f32, shape=(self.batch_size, nmax))
        self.s_old_f = ti.field(dtype=ti.f32, shape=(self.batch_size, nmax))
        self.tmpvec1_f = ti.field(dtype=ti.f32, shape=(self.batch_size, nmax))
        self.tmpvec2_f = ti.field(dtype=ti.f32, shape=(self.batch_size, nmax))
        self.tmpsca1_f = ti.field(dtype=ti.f32, shape=(self.batch_size, ))
        self.tmpsca2_f = ti.field(dtype=ti.f32, shape=(self.batch_size, ))

        self.pos_ones_batch = ti.field(dtype=ti.f32, shape=(self.batch_size, ))
        self.pos_ones_batch.fill(+1.0)
        self.neg_ones_batch = ti.field(dtype=ti.f32, shape=(self.batch_size, ))
        self.neg_ones_batch.fill(-1.0)
        self.zeros_batch = ti.field(dtype=ti.f32, shape=(self.batch_size, ))
        self.zeros_batch.fill(0.0)

        self.delta_0 = ti.field(dtype=ti.f32, shape=(self.batch_size, ))
        self.delta_new = ti.field(dtype=ti.f32, shape=(self.batch_size, ))
        self.delta_old = ti.field(dtype=ti.f32, shape=(self.batch_size, ))
        self.delta_history = ti.field(dtype=ti.f32, shape=(self.batch_size, max_iter))
        """[B, ITER]"""

        self.delta_history_cnt = ti.field(dtype=ti.i32, shape=())
        self.delta_min = ti.field(dtype=ti.f32, shape=(self.batch_size, ))

        self.update_flag = ti.field(dtype=ti.f32, shape=(self.batch_size, ))
        """[B, ] float, 1.0 for update, 0.0 for stop"""
        self.update_flag.fill(1.0)

        self.restart_batch_mask = ti.field(dtype=ti.f32, shape=(self.batch_size, ))
        """[B, ] float, 1.0 for restart, 0.0 for stop"""

        self.iter_cnt = 0
        self.judge_plateau_avg_num = judge_plateau_avg_num

        self.dx_eps = dx_eps
        self.dA_eps = dA_eps

        self.diag_eps = dA_eps
        self.delta_eps = dA_eps * dx_eps ** 2

    @ti.kernel
    def calculate_block_jacobi_inverse_kernel(self, n_field: ti.template(), eps: ti.f32):
        block_dim = ti.static(self.block_dim_jacobi)
        max_n = vec_max_func(n_field, self.batch_size)
        for b, i in ti.ndrange(self.batch_size, (max_n + block_dim - 1) // block_dim):
            if i < (n_field[b] + block_dim - 1) // block_dim:
                mat = ti.Matrix.zero(
                    ti.f32, block_dim, block_dim)
                for j, k in ti.static(ti.ndrange(block_dim, block_dim)):
                    jj = i * block_dim + j
                    kk = i * block_dim + k
                    if jj < n_field[b] and kk < n_field[b]:
                        mat[j, k] = self.p_block_f[b, i, j, k]
                    '''elif jj == kk:
                        mat[j, k] = 1.0'''

                u_mat, s_mat, v_mat = ti.svd(mat)
                s_inv_mat = ti.Matrix.zero(
                    ti.f32, block_dim, block_dim)
                is_singular = False
                for j in ti.static(range(block_dim)):
                    if s_mat[j, j] < eps:
                        s_inv_mat[j, j] = 0.0
                        is_singular = True
                    else:
                        s_inv_mat[j, j] = 1.0 / s_mat[j, j]

                mat_inv = ti.Matrix.zero(
                    ti.f32, block_dim, block_dim)
                if not is_singular:
                    mat_inv = v_mat @ s_inv_mat @ u_mat.transpose()
                else:
                    for j in ti.static(range(block_dim)):
                        if ti.abs(mat[j, j]) > eps:
                            mat_inv[j, j] = 1.0 / mat[j, j]
                        else:
                            mat_inv[j, j] = ti.math.sign(mat[j, j]) / eps

                for j, k in ti.static(ti.ndrange(block_dim, block_dim)):
                    self.p_block_f[b, i, j, k] = mat_inv[j, k]

    def init(self, n_field: ti.Field, incomplete_cholesky_initial_alpha: float = 1e-3):
        """init n, pre-cond and solver's variable."""
        self.n_field.copy_from(n_field)
        self.n_max_field.fill(int(vec_max_kernel(self.n_field, self.batch_size)))
        self.A.compress(self.n_field, self.n_field, True)

        if self.precond == JACOBI_PRECOND:
            self.A.get_diag_kernel(self.tmpvec1_f, self.n_field)
            num_div_vec_batch_kernel(self.batch_size, self.p_f, self.pos_ones_batch, self.tmpvec1_f,
                                     self.n_field, self.diag_eps)
        elif self.precond == INCOMPLETE_CHOLESKY_PRECOND:
            icd_path = os.path.join(os.path.dirname(os.path.abspath(
                __file__)), "../bin/lib_incomplete_cholesky_decomposition.so")

            different_cdll_file_name = []
            for b in range(self.batch_size):
                fn = ".".join(os.path.basename(icd_path).split(".")[:-1]) + str(b).zfill(len(str(self.batch_size - 1))) + ".so"
                different_cdll_file_name.append(os.path.join(os.path.dirname(icd_path), ".tmp", fn))

                os.makedirs(os.path.dirname(different_cdll_file_name[-1]), exist_ok=True)
                shutil.copy(icd_path, different_cdll_file_name[-1])
            self.cdlls = [ctypes.cdll.LoadLibrary(different_cdll_file_name[b]) for b in range(self.batch_size)]

            A_triplet_num_batch = self.A.n_triplet.to_numpy()
            A_np_batch = self.A.triplet.to_numpy()
            for b, A_triplet_num in enumerate(A_triplet_num_batch):
                A_np_modified = {}
                A_np_modified["row"] = A_np_batch["row"][b, ...].astype(np.int32)
                A_np_modified["column"] = A_np_batch["column"][b, ...].astype(np.int32)
                A_np_modified["value"] = A_np_batch["value"][b, ...].astype(np.float32)

                A_row_ptr = numpy_array_to_ptr(A_np_modified["row"], ctypes.c_int32)
                A_col_ptr = numpy_array_to_ptr(A_np_modified["column"], ctypes.c_int32)
                A_val_ptr = numpy_array_to_ptr(A_np_modified["value"], ctypes.c_float)

                self.cdlls[b].IncompleteCholeskyDecomposition(
                    A_row_ptr, A_col_ptr, A_val_ptr,
                    ctypes.c_int32(A_triplet_num), ctypes.c_int32(self.n_field[b]),
                    ctypes.c_float(incomplete_cholesky_initial_alpha)) # this is wrong because we 
        elif self.precond == BLOCK_JACOBI_PRECOND:
            self.A.get_block_diag_kernel(self.p_block_f, self.block_dim_jacobi)
            self.calculate_block_jacobi_inverse_kernel(self.n_field, self.diag_eps)
        else:
            raise NotImplementedError

        self.iter_cnt = 0

    def precond_mul_vec(self, batch_mask: ti.Field, ans: ti.Field, vec: ti.Field):
        """
        precond @ vec = ans
        """
        if self.precond == JACOBI_PRECOND:
            vec_mul_vec_batch_kernel(self.batch_size, batch_mask, ans, vec, self.p_f, self.n)
        elif self.precond == INCOMPLETE_CHOLESKY_PRECOND:
            n_np = self.n_field.to_numpy()
            ans_arr = np.zeros(shape=(self.batch_size, self.nmax), dtype=np.float32)
            vec_arr = vec.to_numpy()
            batch_mask_arr = batch_mask.to_numpy()
            for b in range(self.batch_size):
                if batch_mask_arr[b] == 0.0:
                    continue
                _ans_arr = ans_arr[b, :n_np[b]]
                _vec_arr = vec_arr[b, :n_np[b]]
                ans_arr_ptr = numpy_array_to_ptr(_ans_arr, ctype=ctypes.c_float)
                vec_arr_ptr = numpy_array_to_ptr(_vec_arr, ctype=ctypes.c_float)
                self.cdlls[b].SolveLLTx(ans_arr_ptr, vec_arr_ptr, self.n)
                ans_arr[b, :n_np[b]] = _ans_arr

            ans.from_numpy(ans_arr)
        elif self.precond == BLOCK_JACOBI_PRECOND:
            block_mul_vec_batch_kernel(self.batch_size, batch_mask, ans, self.p_block_f, vec, self.n_field, self.block_dim_jacobi)
        else:
            raise NotImplementedError

    @ti.kernel
    def restart_stage_1_kernel(self, batch_mask: ti.template()):
        for batch_idx, i in ti.ndrange(self.batch_size, self.n_max_field[None]):
            if i < self.n_field[batch_idx] and batch_mask[batch_idx] != 0.0:
                self.x_f[batch_idx, i] = self.x_best_f[batch_idx, i]
                self.s_f[batch_idx, i] = 0.0
        self.A.mul_vec_func(batch_mask, self.r_f, self.x_f, self.n_field)
        vec_add_vec_batch_func(self.batch_size, batch_mask, self.r_f, self.pos_ones_batch, self.b, self.neg_ones_batch, self.r_f, self.n_field)
        vec_mul_vec_batch_func(self.batch_size, batch_mask, self.r_f, self.r_f, self.con_f, self.n_field)

    @ti.kernel
    def restart_stage_2_kernel(self, batch_mask: ti.template(), CG_relative_tol:ti.f32) -> bool:

        vec_mul_vec_batch_func(self.batch_size, batch_mask, self.c_f, self.c_f, self.con_f, self.n_field)
        vec_dot_vec_kahan_batch_func(self.batch_size, batch_mask, self.c_f, self.r_f, self.delta_new, self.tmpsca1_f, self.n_field)

        for _b in range(self.batch_size):
            if self.delta_new[_b] < self.eps_f[_b] or \
                self.delta_new[_b] < self.delta_0[_b] * CG_relative_tol ** 2:
                self.update_flag[_b] = 0.0

        all_is_converge = True
        for _b in range(self.batch_size):
            if self.update_flag[_b] != 0.0:
                all_is_converge = False
        return all_is_converge
        
    def restart(self, batch_mask: ti.Field, CG_relative_tol: float) -> bool:
        self.restart_stage_1_kernel(batch_mask)
        self.precond_mul_vec(batch_mask, self.c_f, self.r_f)
        return self.restart_stage_2_kernel(batch_mask, CG_relative_tol)

    def iter_init(self, CG_relative_tol: float) -> bool:
        batch_mask = self.update_flag
        # calculate delta_0
        vec_mul_vec_batch_kernel(self.batch_size, batch_mask, self.tmpvec1_f, self.b, self.con_f, self.n_field)
        self.precond_mul_vec(batch_mask, self.tmpvec2_f, self.tmpvec1_f)
        vec_dot_vec_kahan_batch_kernel(self.batch_size, batch_mask, self.tmpvec2_f, self.tmpvec1_f, self.delta_0, self.tmpsca1_f, self.n_field)
        self.delta_min.copy_from(self.delta_0)

        return self.restart(batch_mask, CG_relative_tol)

    @ti.kernel
    def iter_step_stage_1_kernel(self) -> ti.types.vector(2, bool):
        """
        ```
        [PREV] q := A @ c
        q := con * q
        c_dot_q := c^T @ q
        if c_dot_q > eps_f:
            alpha := delta_new / c_dot_q
            is_converge := False
        else:
            alpha := 0.0
            is_converge := False
        x := x + alpha * c
        r := r - alpha * q
        s_old := s
        return is_converge
        [NEXT] s := M^-1 @ r (M^-1 @ A ~ I)
        ```
        """
        self.A.mul_vec_func(self.pos_ones_batch, self.q_f, self.c_f, self.n_field)

        # tmpsca1_f = c_dot_q, tmpsca2_f = res
        self.tmpsca1_f.fill(0.0)
        self.tmpsca2_f.fill(0.0)
        for batch_idx, i in ti.ndrange(self.batch_size, self.n_max_field[None]):
            if i < self.n_field[batch_idx]:
                self.q_f[batch_idx, i] *= self.con_f[batch_idx, i]
                mul = self.c_f[batch_idx, i] * self.q_f[batch_idx, i]
                old_sum = ti.atomic_add(self.tmpsca1_f[batch_idx], mul)
                if ti.abs(old_sum) > ti.abs(mul):
                    self.tmpsca2_f[batch_idx] += ((old_sum + mul) - old_sum) - mul
                else:
                    self.tmpsca2_f[batch_idx] += ((old_sum + mul) - mul) - old_sum

        all_is_converge = True
        need_restart = False
        for batch_idx in range(self.batch_size):
            self.tmpsca1_f[batch_idx] -= self.tmpsca2_f[batch_idx]
            if self.tmpsca1_f[batch_idx] < self.eps_f[batch_idx]:
                self.update_flag[batch_idx] = 0.0

            # tmpsca2_f = alpha
            else:
                self.tmpsca2_f[batch_idx] = self.delta_new[batch_idx] / self.tmpsca1_f[batch_idx]

            if self.tmpsca1_f[batch_idx] < self.delta_eps and self.print_warning:
                print("[WARNING] c_dot_q:{} < delta_eps:{}, A_eps or x_eps may be too large, or eps_f may be too small.".format(
                    self.tmpsca1_f[batch_idx], self.delta_eps))
                self.update_flag[batch_idx] = 0.0
            
            if self.update_flag[batch_idx] != 0.0:
                all_is_converge = False
            if self.restart_batch_mask[batch_idx] != 0.0:
                need_restart = True

        for batch_idx, i in ti.ndrange(self.batch_size, self.n_max_field[None]):
            if i < self.n_field[batch_idx] and self.update_flag[batch_idx] != 0.0:
                self.x_f[batch_idx, i] += self.tmpsca2_f[batch_idx] * self.c_f[batch_idx, i]
                self.r_f[batch_idx, i] -= self.tmpsca2_f[batch_idx] * self.q_f[batch_idx, i]
                self.s_old_f[batch_idx, i] = self.s_f[batch_idx, i]
            
        return all_is_converge, need_restart

    @ti.kernel
    def iter_step_stage_2_kernel(self, CG_relative_tol: ti.f32, restart_threshold: ti.f32) -> ti.types.vector(2, bool):
        """
        ```
        [PREV] s := M^-1 @ r (M^-1 @ A ~ I)
        delta_old := delta_new
        tmp1 := s - s_old
        delta_new := tmp1^T @ r
        beta := delta_new / max(delta_old, eps)
        if delta_new < delta_minimum:
            x_best := x
        c := s + beta * c
        c := con * c
        return delta_new < max(eps_f, delta_0 * rel_eps ^ 2) or plateau
        [NEXT] q := A @ c
        ```
        """
        old_cnt = ti.atomic_add(self.delta_history_cnt[None], 1)
        start = ti.max(0, old_cnt - self.judge_plateau_avg_num)
        end = old_cnt

        for batch_idx in range(self.batch_size):
            self.delta_old[batch_idx] = self.delta_new[batch_idx]

        self.delta_new.fill(0.0)
        self.tmpsca2_f.fill(0.0) # res
        for batch_idx, i in ti.ndrange(self.batch_size, self.n_max_field[None]):
            if i < self.n_field[batch_idx] and self.update_flag[batch_idx] != 0.0:
                mul = (self.s_f[batch_idx, i] - self.s_old_f[batch_idx, i]) * self.r_f[batch_idx, i]
                old_sum = ti.atomic_add(self.delta_new[batch_idx], mul)
                if ti.abs(old_sum) > ti.abs(mul):
                    self.tmpsca2_f[batch_idx] += ((old_sum + mul) - old_sum) - mul
                else:
                    self.tmpsca2_f[batch_idx] += ((old_sum + mul) - mul) - old_sum

        # vec_add_vec_batch_func(self.batch_size, self.pos_ones_batch, self.tmpvec1_f, self.pos_ones_batch, self.s_f, self.neg_ones_batch, self.s_old_f, self.n_field)
        # vec_dot_vec_kahan_batch_func(self.batch_size, self.update_flag, self.r_f, self.tmpvec1_f, self.delta_new, self.tmpsca1_f, self.n_field)
        for batch_idx in range(self.batch_size):
            if self.update_flag[batch_idx] != 0.0:
                self.delta_new[batch_idx] -= self.tmpsca2_f[batch_idx]
                self.delta_history[batch_idx, old_cnt] = self.delta_new[batch_idx]
            
            ti.atomic_min(self.delta_min[batch_idx], self.delta_new[batch_idx])
            self.tmpsca1_f[batch_idx] = 0.0
            self.tmpsca2_f[batch_idx] = self.delta_new[batch_idx] / \
            ti.max(self.delta_old[batch_idx], self.delta_eps) # tmpsca2_f = beta

            if self.delta_old[batch_idx] < self.delta_eps and self.print_warning:
                print("[WARNING] delta_old:{} < delta_eps:{}, A_eps or x_eps may be too large, or eps_f may be too small.".format(
                    self.delta_old[batch_idx], self.delta_eps))
                self.update_flag[batch_idx] = 0.0
                
        for batch_idx, i in ti.ndrange(self.batch_size, (start, end)):
            self.tmpsca1_f[batch_idx] += self.delta_history[batch_idx, i] # tmpsca1_f = delta_sum

        for batch_idx, i in ti.ndrange(self.batch_size, self.n_max_field[None]):
            if i < self.n_field[batch_idx]:
                if self.delta_min[batch_idx] == self.delta_new[batch_idx]:
                    self.x_best_f[batch_idx, i] = self.x_f[batch_idx, i]
                self.c_f[batch_idx, i] = (self.s_f[batch_idx, i] + self.tmpsca2_f[batch_idx] * self.c_f[batch_idx, i]) * self.con_f[batch_idx, i]
            
        all_is_converge = True
        need_restart = False
        for batch_idx in range(self.batch_size):
            if self.delta_new[batch_idx] > self.delta_old[batch_idx] and \
                self.delta_new[batch_idx] > restart_threshold * self.delta_min[batch_idx]:
                self.restart_batch_mask[batch_idx] = 1.0
                need_restart = True
            else:
                self.restart_batch_mask[batch_idx] = 0.0

            if (self.delta_new[batch_idx] < self.eps_f[batch_idx] or
                self.delta_new[batch_idx] < self.delta_0[batch_idx] * CG_relative_tol ** 2) or \
                (self.delta_new[batch_idx] * (end - start) > self.tmpsca1_f[batch_idx] and
                old_cnt >= self.judge_plateau_avg_num and self.restart_batch_mask[batch_idx] == 0.0):
                self.update_flag[batch_idx] = 0.0
            if self.update_flag[batch_idx] != 0.0:
                all_is_converge = False
            
        return all_is_converge, need_restart

    def all_is_converge(self) -> bool:
        return np.sum(self.update_flag.to_numpy()) < 0.5 # all of update_flag are zero

    def iter_step(self, CG_relative_tol: float, restart_threshold: float) -> Tuple[bool, bool]:
        # self.clock.start_clock("A")
        # self.A.mul_vec_kernel(self.pos_ones_batch, self.q_f, self.c_f, self.n_field)
        # self.clock.end_clock("A")
        # self.clock.start_clock("B")
        all_is_converge, need_restart = self.iter_step_stage_1_kernel()
        # self.clock.end_clock("B")
        if not all_is_converge:
            # self.clock.start_clock("C")
            self.precond_mul_vec(self.pos_ones_batch, self.s_f, self.r_f)
            # self.clock.end_clock("C")
            # self.clock.start_clock("D")
            all_is_converge, need_restart = self.iter_step_stage_2_kernel(CG_relative_tol, restart_threshold)
            # self.clock.end_clock("D")
            return all_is_converge, need_restart # converge = not update
        else:
            return True, False

    def solve_python(self, max_iter: int, init_zero: bool, x0: ti.Field, update_flag: ti.Field, CG_relative_tol: float, restart_threshold: bool) -> np.ndarray:
        assert max_iter >= 1

        self.update_flag.copy_from(update_flag)

        if init_zero:
            self.x_f.fill(0.0)
            vec_mul_vec_batch_kernel(self.batch_size, self.pos_ones_batch, self.x_f, self.x_f, self.con_f, self.n_field)
            self.x_best_f.copy_from(self.x_f)
        else:
            vec_copy_batch_kernel(self.batch_size, self.x_f, x0, self.n_field)
            vec_mul_vec_batch_kernel(self.batch_size, self.pos_ones_batch, self.x_f, self.x_f, self.con_f, self.n_field)
            self.x_best_f.copy_from(self.x_f)

        self.delta_history.fill(0.0)
        self.delta_history_cnt[None] = 0
        all_is_converge = self.iter_init(CG_relative_tol)

        if not all_is_converge:
            remain_iter = min(max_iter, np.max(self.n_field.to_numpy()))

            while remain_iter >= 1:
                all_is_converge, need_restart = self.iter_step(CG_relative_tol, restart_threshold)

                remain_iter -= 1
                self.iter_cnt += 1

                if (not all_is_converge) and need_restart:
                    all_is_converge = self.restart(self.restart_batch_mask, CG_relative_tol)

                if all_is_converge:
                    break

        return self.x_best_f.to_numpy()

    def solve(self, max_iter: int, init_zero: bool, x0: ti.Field, update_flag: ti.Field, CG_relative_tol: float, restart_threshold: bool = 1e1) -> np.ndarray:
        """return the solution of Ax = b."""
        return self.solve_python(max_iter, init_zero, x0, update_flag, CG_relative_tol, restart_threshold)
