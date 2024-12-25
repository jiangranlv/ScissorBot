#include <iostream>
#include <cassert>
#include <sys/time.h>
#include "Eigen/Sparse"

typedef int32_t integer;
typedef float real;

#ifndef ToReal
#define ToReal real
#endif

typedef Eigen::SparseMatrix<real, Eigen::ColMajor> SparseMatrixXr;
typedef Eigen::Matrix<real, Eigen::Dynamic, 1> VectorXr;
typedef Eigen::Triplet<real> Triplet;

SparseMatrixXr lower_triangular;

void CheckCondition(bool cond, std::string err_loc, std::string err_msg)
{
    if (!cond)
        std::cout << "[ERROR] in " << err_loc << " , " << err_msg << std::endl;
}

SparseMatrixXr FromDiagonal(const VectorXr &vec)
{
    const integer len = vec.rows();
    SparseMatrixXr m(len, len);

    for (int k = 0; k < vec.outerSize(); ++k)
    {
        for (VectorXr::InnerIterator it(vec, k); it; ++it)
        {
            m.insert(it.row(), it.row()) = it.value();
        }
    }

    return m;
}

bool IsClose(real x, real y, real eps, real val)
{
    return std::abs(x - y - val) <= eps;
}

SparseMatrixXr ComputeAnalyticalPreconditioner(const SparseMatrixXr &lhs, integer n, real initial_alpha = 1e-3)
{
    const std::string error_location = "ComputeAnalyticalPreconditioner";
    CheckCondition(lhs.isCompressed(), error_location, "We expect lhs is compressed.");

    SparseMatrixXr lhs_lower = lhs.triangularView<Eigen::Lower>();
    // The internal data structure of lhs_lower:
    // - valuePtr: all nonzeros (size: nonzero numbers).
    // - innerIndexPtr: row indices of nonzeros (size: nonzero numbers). Always sorted in increasing order within each column.
    // - outerIndexPtr: OuterStarts[i] is the index in the previous two arrays of the first nonzeros in column i (size: column number).
    // - innerNonZeroPtr: the number of nonzeros in each column (size: column number).
    // const integer n = this->matrix_size();
    CheckCondition(lhs_lower.rows() == lhs_lower.cols() && static_cast<integer>(lhs_lower.rows()) == n, error_location, "Incorrect size for lhs.");

    // Construct inner_nonzeros.
    for (integer i = 0; i < n; ++i)
    {
        CheckCondition(lhs_lower.outerIndexPtr()[i] != lhs_lower.outerIndexPtr()[i + 1], error_location, "Some column in lhs is all zero.");
        CheckCondition(lhs_lower.innerIndexPtr()[lhs_lower.outerIndexPtr()[i]] == i, error_location, "lhs does not have some diagonal elements.");
        const real val = lhs_lower.valuePtr()[lhs_lower.outerIndexPtr()[i]];
        CheckCondition(val > 0, error_location, "lhs has expect positive diagonals but obtain " + std::to_string(val) + ".");
    }

    // Implementing Alg. 3.1 from the paper "Incomplete Cholesky Factorizations with Limited Memory".
    // Compute D = diag(|lhs|_2).
    VectorXr D = VectorXr::Zero(n);
    for (integer i = 0; i < n; ++i)
    {
        for (integer j = lhs.outerIndexPtr()[i]; j < lhs.outerIndexPtr()[i + 1]; ++j)
        {
            const real v = lhs.valuePtr()[j];
            D(i) += v * v;
        }
    }
    D = D.cwiseSqrt();
    // D = diag(|lhs|_2) now.
    const VectorXr D_sqrt = D.cwiseSqrt();
    const VectorXr inv_D_sqrt = D_sqrt.cwiseInverse();

    // Compute lhs_hat = D^{-1/2} * lhs * D^{-1/2}.
    for (integer i = 0; i < n; ++i)
    {
        for (integer j = lhs_lower.outerIndexPtr()[i]; j < lhs_lower.outerIndexPtr()[i + 1]; ++j)
        {
            lhs_lower.valuePtr()[j] *= inv_D_sqrt(i) * inv_D_sqrt(lhs_lower.innerIndexPtr()[j]);
        }
    }
    // Now lhs_lower = lhs_hat.
    const real alpha_s(initial_alpha); // Suggested by the paper.
    real alpha_i = 0;
    integer alpha_increase_cnt = 0;
    while (true)
    {
        // Try factorizing lhs_lower + alpha_i * I.
        SparseMatrixXr lhs_lower_shift = lhs_lower + alpha_i * FromDiagonal(VectorXr::Ones(n));

        bool success = true;
        for (integer j = 0; j < n; ++j)
        {
            // If the diagonal becomes negative, break.
            real ljj = lhs_lower_shift.valuePtr()[lhs_lower_shift.outerIndexPtr()[j]];
            if (ljj <= 0)
            {
                success = false;
                break;
            }
            ljj = std::sqrt(ljj);
            lhs_lower_shift.valuePtr()[lhs_lower_shift.outerIndexPtr()[j]] = ljj;

            const real inv_ljj = ToReal(1) / ljj;
            for (integer k = lhs_lower_shift.outerIndexPtr()[j] + 1; k < lhs_lower_shift.outerIndexPtr()[j + 1]; ++k)
            {
                lhs_lower_shift.valuePtr()[k] *= inv_ljj;
            }

            for (integer kj2 = lhs_lower_shift.outerIndexPtr()[j] + 1; kj2 < lhs_lower_shift.outerIndexPtr()[j + 1]; ++kj2)
            {
                const integer j2 = lhs_lower_shift.innerIndexPtr()[kj2];
                const real lj2j = lhs_lower_shift.valuePtr()[kj2];
                integer i_cur_cnt = lhs_lower_shift.outerIndexPtr()[j];
                integer i_cur = lhs_lower_shift.innerIndexPtr()[i_cur_cnt];
                // i_cur == j and < the very first i below for sure.
                for (integer k = lhs_lower_shift.outerIndexPtr()[j2]; k < lhs_lower_shift.outerIndexPtr()[j2 + 1]; ++k)
                {
                    const integer i = lhs_lower_shift.innerIndexPtr()[k];
                    // lhs_lower_shift[i][j2] -= lhs_lower_shift[i][j] * lhs_lower_shift[j2][j], i.e.,
                    // lhs_lower_shift[i][j2] -= lhs_lower_shift[i][j] * lj2j.
                    while (i_cur < i && i_cur_cnt < lhs_lower_shift.outerIndexPtr()[j + 1] - 1)
                    {
                        ++i_cur_cnt;
                        i_cur = lhs_lower_shift.innerIndexPtr()[i_cur_cnt];
                    }
                    if (i_cur == i)
                    {
                        // lhs_lower_shift[i][j2] -= lhs_lower_shift[i][j] * lj2j.
                        lhs_lower_shift.valuePtr()[k] -=
                            lhs_lower_shift.valuePtr()[i_cur_cnt] * lj2j;
                    }
                    if (i_cur_cnt == lhs_lower_shift.outerIndexPtr()[j + 1] - 1 && i_cur <= i)
                        break;
                }
            }
        }

        if (success)
        {
            // Sanity check:
            /*
            const SparseMatrixXr L = lhs_lower_shift;
            const SparseMatrixXr Lt = SparseMatrixXr(L.transpose());
            SparseMatrixXr LLt = L * Lt;
            const SparseMatrixXr A = lhs_lower + alpha_i * FromDiagonal(VectorXr::Ones(n));
            // L should have the same sparsity as A.
            CheckCondition(L.nonZeros() == A.nonZeros(), error_location, "L and A have different number of nonzeros.");
            // - innerIndexPtr: row indices of nonzeros (size: nonzero numbers). Always sorted in increasing order within each column.
            // - outerIndexPtr: OuterStarts[i] is the index in the previous two arrays of the first nonzeros in column i (size: column number).
            const integer nonzeros = static_cast<integer>(L.nonZeros());
            for (integer i = 0; i < nonzeros; ++i)
            {
                CheckCondition(L.innerIndexPtr()[i] == A.innerIndexPtr()[i], error_location, "L and A disagrees with inner indices.");
            }
            for (integer i = 0; i < n; ++i)
            {
                CheckCondition(L.outerIndexPtr()[i] == A.outerIndexPtr()[i], error_location, "L and A disagrees with outer indices.");
            }
            // L * Lt should agree with A on A's nonzeros.
            for (integer j = 0; j < n; ++j)
            {
                for (SparseMatrixXr::InnerIterator it(A, j); it; ++it)
                {
                    const integer i = it.row();
                    const real val = it.value();
                    // A(i, j) = val and is a nonzero element.
                    // Does L * Lt contain this element?
                    const integer nn_before = LLt.nonZeros();
                    const real LLt_val = LLt.coeffRef(i, j);
                    const integer nn_after = LLt.nonZeros();
                    CheckCondition(nn_before == nn_after, error_location, "A(i, j) does not exist in LLt.");
                    CheckCondition(IsClose(val, LLt_val, 1e-6, 0), error_location, "LLt (" + std::to_string(LLt_val) + ") and A (" + std::to_string(val) + ") + disagree at (" + std::to_string(i) + ", " + std::to_string(j) + ").");
                }
            }*/

            // Return.
            // L * Lt = D^{-1/2} * lhs * D^{-1/2}.
            // D^{1/2} * L * Lt * D^{1/2} = lhs.
            // This means each row L[i] should be multiplied by D^{1/2}(i).
            for (integer j = 0; j < n; ++j)
            {
                for (integer k = lhs_lower_shift.outerIndexPtr()[j]; k < lhs_lower_shift.outerIndexPtr()[j + 1]; ++k)
                {
                    lhs_lower_shift.valuePtr()[k] *= D_sqrt(lhs_lower_shift.innerIndexPtr()[k]);
                }
            }
            std::cout << "[IC] increase alpha cnt" << alpha_increase_cnt << std::endl;
            return lhs_lower_shift;
        }
        else
        {
            // Update alpha_i.
            alpha_i = std::max(alpha_i * 2, alpha_s);
            alpha_increase_cnt += 1;
        }
    }
    CheckCondition(false, error_location, "Jumped out of the main loop unexpectedly.");
    return SparseMatrixXr();
}

extern "C" void IncompleteCholeskyDecomposition(integer A_row[], integer A_col[], real A_val[], integer A_triplet_num, integer n, real initial_alpha = 1e-3)
{
    assert(A_triplet_num >= 1);
    assert(n >= 1);

    SparseMatrixXr A(n, n);
    std::vector<Triplet> tripletList;
    tripletList.reserve(A_triplet_num);
    for (integer i = 0; i < A_triplet_num; i++)
    {
        integer r = A_row[i];
        integer c = A_col[i];
        real v = A_val[i];
        assert(0 <= r && r < n && 0 <= c && c < n);
        tripletList.push_back(Triplet(r, c, v));
    }

    A.setFromTriplets(tripletList.begin(), tripletList.end());
    A.makeCompressed();

    lower_triangular = ComputeAnalyticalPreconditioner(A, n, initial_alpha);
}

extern "C" integer GetLowerTriangular(integer L_row[], integer L_col[], real L_val[], integer L_triplet_max, integer n)
{
    assert(L_triplet_max >= 1);
    assert(n >= 1);

    integer L_triplet_num = 0;
    for (int k = 0; k < lower_triangular.outerSize(); ++k)
    {
        for (SparseMatrixXr::InnerIterator it(lower_triangular, k); it; ++it)
        {
            assert(L_triplet_num < L_triplet_max);

            int r = it.row();
            int c = it.col();
            float v = it.value();

            L_row[L_triplet_num] = r;
            L_col[L_triplet_num] = c;
            L_val[L_triplet_num] = v;

            L_triplet_num += 1;
        }
    }
    return L_triplet_num;
}

extern "C" void SolveLLTx(real x[], real y[], integer n)
{
    CheckCondition(static_cast<integer>(n) == lower_triangular.rows() && static_cast<integer>(n) == lower_triangular.cols(), "SolveLLTx", "Incompatible sizes between rhs and lhs.");

    VectorXr rhs(n, 1);
    for (integer i = 0; i < n; ++i)
        rhs(i, 0) = y[i];

    const VectorXr mhs = lower_triangular.triangularView<Eigen::Lower>().solve(rhs);             // Solve L @ mhs = rhs.
    const VectorXr lhs = lower_triangular.transpose().triangularView<Eigen::Upper>().solve(mhs); // Solve LT @ lhs = mhs.

    for (integer i = 0; i < n; ++i)
        x[i] = lhs(i, 0);
}