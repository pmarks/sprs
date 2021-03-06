extern crate num_traits;
///! Cholesky factorization module.
///!
///! Contains LDLT decomposition methods.
///!
///! This decomposition operates on symmetric positive definite matrices,
///! and is written `A = L D L` where L is lower triangular and D is diagonal.
///! It is closely related to the Cholesky decomposition, but is often more
///! numerically stable and can work on some indefinite matrices.
///!
///! The easiest way to use this API is to create a `LdlNumeric` instance from
///! a matrix, then use the `LdlNumeric::solve` method.
///!
///! It is possible to update a decomposition if the sparsity structure of a
///! matrix does not change. In that case the `LdlNumeric::update` method can
///! be used.
///!
///! When only the sparsity structure of a matrix is known, it is possible
///! to precompute part of the factorization by using the `LdlSymbolic` struct.
///! This struct can the be converted into a `LdlNumeric` once the non-zero
///! values are known, using the `LdlSymbolic::factor` method.
// This method is adapted from the LDL library by Tim Davis:
//
// LDL Copyright (c) 2005 by Timothy A. Davis.  All Rights Reserved.
//
// LDL License:
//
//     Your use or distribution of LDL or any modified version of
//     LDL implies that you agree to this License.
//
//     This library is free software; you can redistribute it and/or
//     modify it under the terms of the GNU Lesser General Public
//     License as published by the Free Software Foundation; either
//     version 2.1 of the License, or (at your option) any later version.
//
//     This library is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//     Lesser General Public License for more details.
//
//     You should have received a copy of the GNU Lesser General Public
//     License along with this library; if not, write to the Free Software
//     Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301
//     USA
//
//     Permission is hereby granted to use or copy this program under the
//     terms of the GNU LGPL, provided that the Copyright, this License,
//     and the Availability of the original version is retained on all copies.
//     User documentation of any code that uses this code or any modified
//     version of this code must cite the Copyright, this License, the
//     Availability note, and "Used by permission." Permission to modify
//     the code and to distribute modified code is granted, provided the
//     Copyright, this License, and the Availability note are retained,
//     and a notice that the code was modified is included.
extern crate sprs;

use std::ops::Deref;
use std::ops::IndexMut;

use num_traits::Num;

use sprs::errors::SprsError;
use sprs::indexing::SpIndex;
use sprs::linalg;
use sprs::stack::DStack;
use sprs::{is_symmetric, CsMatViewI, PermOwnedI, Permutation};

pub enum SymmetryCheck {
    CheckSymmetry,
    DontCheckSymmetry,
}

/// Structure to compute and hold a symbolic LDLT decomposition
#[derive(Debug, Clone)]
pub struct LdlSymbolic<I> {
    colptr: Vec<I>,
    parents: linalg::etree::ParentsOwned,
    nz: Vec<I>,
    flag_workspace: Vec<I>,
    perm: Permutation<I, Vec<I>>,
}

/// Structure to hold a numeric LDLT decomposition
#[derive(Debug, Clone)]
pub struct LdlNumeric<N, I> {
    symbolic: LdlSymbolic<I>,
    l_indices: Vec<I>,
    l_data: Vec<N>,
    diag: Vec<N>,
    y_workspace: Vec<N>,
    pattern_workspace: DStack<I>,
}

impl<I: SpIndex> LdlSymbolic<I> {
    /// Compute the symbolic LDLT of the given matrix
    ///
    /// # Panics
    ///
    /// * if mat is not symmetric
    pub fn new<N>(mat: CsMatViewI<N, I>) -> LdlSymbolic<I>
    where
        N: Copy + PartialEq,
    {
        assert_eq!(mat.rows(), mat.cols());
        let perm: Permutation<I, Vec<I>> = Permutation::identity(mat.rows());
        LdlSymbolic::new_perm(mat, perm)
    }

    /// Compute the symbolic decomposition L D L^T = P A P^T
    /// where P is a permutation matrix.
    ///
    /// Using a good permutation matrix can reduce the non-zero count in L,
    /// thus making the decomposition and the solves faster.
    ///
    /// # Panics
    ///
    /// * if mat is not symmetric
    pub fn new_perm<N>(
        mat: CsMatViewI<N, I>,
        perm: PermOwnedI<I>,
    ) -> LdlSymbolic<I>
    where
        N: Copy + PartialEq,
        I: SpIndex,
    {
        let n = mat.cols();
        assert!(mat.rows() == n, "matrix should be square");
        let mut l_colptr = vec![I::zero(); n + 1];
        let mut parents = linalg::etree::ParentsOwned::new(n);
        let mut l_nz = vec![I::zero(); n];
        let mut flag_workspace = vec![I::zero(); n];
        ldl_symbolic(
            mat,
            &perm,
            &mut l_colptr,
            parents.view_mut(),
            &mut l_nz,
            &mut flag_workspace,
            SymmetryCheck::CheckSymmetry,
        );

        LdlSymbolic {
            colptr: l_colptr,
            parents: parents,
            nz: l_nz,
            flag_workspace: flag_workspace,
            perm: perm,
        }
    }

    /// The size of the linear system associated with this decomposition
    #[inline]
    pub fn problem_size(&self) -> usize {
        self.parents.nb_nodes()
    }

    /// The number of non-zero entries in L
    #[inline]
    pub fn nnz(&self) -> usize {
        let n = self.problem_size();
        self.colptr[n].index()
    }

    /// Compute the numerical decomposition of the given matrix.
    pub fn factor<N>(
        self,
        mat: CsMatViewI<N, I>,
    ) -> Result<LdlNumeric<N, I>, SprsError>
    where
        N: Copy + Num + PartialOrd,
    {
        let n = self.problem_size();
        let nnz = self.nnz();
        let l_indices = vec![I::zero(); nnz];
        let l_data = vec![N::zero(); nnz];
        let diag = vec![N::zero(); n];
        let y_workspace = vec![N::zero(); n];
        let pattern_workspace = DStack::with_capacity(n);
        let mut ldl_numeric = LdlNumeric {
            symbolic: self,
            l_indices: l_indices,
            l_data: l_data,
            diag: diag,
            y_workspace: y_workspace,
            pattern_workspace: pattern_workspace,
        };
        ldl_numeric.update(mat).map(|_| ldl_numeric)
    }
}

impl<N, I: SpIndex> LdlNumeric<N, I> {
    /// Compute the numeric LDLT decomposition of the given matrix.
    ///
    /// # Panics
    ///
    /// * if mat is not symmetric
    pub fn new(mat: CsMatViewI<N, I>) -> Result<Self, SprsError>
    where
        N: Copy + Num + PartialOrd,
    {
        let symbolic = LdlSymbolic::new(mat.view());
        symbolic.factor(mat)
    }

    /// Compute the numeric decomposition L D L^T = P^T A P
    /// where P is a permutation matrix.
    ///
    /// Using a good permutation matrix can reduce the non-zero count in L,
    /// thus making the decomposition and the solves faster.
    ///
    /// # Panics
    ///
    /// * if mat is not symmetric
    pub fn new_perm(
        mat: CsMatViewI<N, I>,
        perm: PermOwnedI<I>,
    ) -> Result<Self, SprsError>
    where
        N: Copy + Num + PartialOrd,
    {
        let symbolic = LdlSymbolic::new_perm(mat.view(), perm);
        symbolic.factor(mat)
    }

    /// Update the decomposition with the given matrix. The matrix must
    /// have the same non-zero pattern as the original matrix, otherwise
    /// the result is unspecified.
    pub fn update(&mut self, mat: CsMatViewI<N, I>) -> Result<(), SprsError>
    where
        N: Copy + Num + PartialOrd,
    {
        ldl_numeric(
            mat.view(),
            &self.symbolic.colptr,
            self.symbolic.parents.view(),
            &self.symbolic.perm,
            &mut self.symbolic.nz,
            &mut self.l_indices,
            &mut self.l_data,
            &mut self.diag,
            &mut self.y_workspace,
            &mut self.pattern_workspace,
            &mut self.symbolic.flag_workspace,
        )
    }

    /// Solve the system A x = rhs
    pub fn solve<'a, V>(&self, rhs: &V) -> Vec<N>
    where
        N: 'a + Copy + Num,
        V: Deref<Target = [N]>,
    {
        let mut x = &self.symbolic.perm * &rhs[..];
        let l = self.l_view();
        ldl_lsolve(&l, &mut x);
        linalg::diag_solve(&self.diag, &mut x);
        ldl_ltsolve(&l, &mut x);
        let pinv = self.symbolic.perm.inv();
        &pinv * &x
    }

    fn l_view(&self) -> CsMatViewI<N, I> {
        let n = self.symbolic.problem_size();
        // CsMat invariants are guaranteed by the LDL algorithm
        unsafe {
            CsMatViewI::new_view_raw(
                sprs::CSC,
                (n, n),
                self.symbolic.colptr.as_ptr(),
                self.l_indices.as_ptr(),
                self.l_data.as_ptr(),
            )
        }
    }

    /// The size of the linear system associated with this decomposition
    #[inline]
    pub fn problem_size(&self) -> usize {
        self.symbolic.problem_size()
    }

    /// The number of non-zero entries in L
    #[inline]
    pub fn nnz(&self) -> usize {
        self.symbolic.nnz()
    }
}

/// Perform a symbolic LDLt decomposition of a symmetric sparse matrix
pub fn ldl_symbolic<N, I, PStorage>(
    mat: CsMatViewI<N, I>,
    perm: &Permutation<I, PStorage>,
    l_colptr: &mut [I],
    mut parents: linalg::etree::ParentsViewMut,
    l_nz: &mut [I],
    flag_workspace: &mut [I],
    check_symmetry: SymmetryCheck,
) where
    N: Clone + Copy + PartialEq,
    I: SpIndex,
    PStorage: Deref<Target = [I]>,
{
    match check_symmetry {
        SymmetryCheck::DontCheckSymmetry => (),
        SymmetryCheck::CheckSymmetry => {
            if !is_symmetric(&mat) {
                panic!("Matrix is not symmetric")
            }
        }
    }

    let n = mat.rows();

    let outer_it = mat.outer_iterator_perm(perm.view());
    // compute the elimination tree of L
    for (k, (_, vec)) in outer_it.enumerate() {
        flag_workspace[k] = I::from_usize(k); // this node is visited
        parents.set_root(k);
        l_nz[k] = I::zero();

        for (inner_ind, _) in vec.iter_perm(perm.inv()) {
            let mut i = inner_ind;

            if i < k {
                while flag_workspace[i].index() != k {
                    parents.uproot(i, k);
                    l_nz[i] += I::one();
                    flag_workspace[i] = I::from_usize(k);
                    i = parents.get_parent(i).expect("uprooted so not a root");
                }
            }
        }
    }

    let mut prev = I::zero();
    for (k, colptr) in (0..n).zip(l_colptr.iter_mut()) {
        *colptr = prev;
        prev += l_nz[k];
    }
    l_colptr[n] = prev;
}

/// Perform numeric LDLT decomposition
///
/// pattern_workspace is a DStack of capacity n
pub fn ldl_numeric<N, I, PStorage>(
    mat: CsMatViewI<N, I>,
    l_colptr: &[I],
    parents: linalg::etree::ParentsView,
    perm: &Permutation<I, PStorage>,
    l_nz: &mut [I],
    l_indices: &mut [I],
    l_data: &mut [N],
    diag: &mut [N],
    y_workspace: &mut [N],
    pattern_workspace: &mut DStack<I>,
    flag_workspace: &mut [I],
) -> Result<(), SprsError>
where
    N: Clone + Copy + PartialEq + Num + PartialOrd,
    I: SpIndex,
    PStorage: Deref<Target = [I]>,
{
    let outer_it = mat.outer_iterator_perm(perm.view());
    for (k, (_, vec)) in outer_it.enumerate() {
        // compute the nonzero pattern of the kth row of L
        // in topological order

        flag_workspace[k] = I::from_usize(k); // this node is visited
        y_workspace[k] = N::zero();
        l_nz[k] = I::zero();
        pattern_workspace.clear_right();

        for (inner_ind, &val) in
            vec.iter_perm(perm.inv()).filter(|&(i, _)| i <= k)
        {
            y_workspace[inner_ind] = y_workspace[inner_ind] + val;
            let mut i = inner_ind;
            pattern_workspace.clear_left();
            while flag_workspace[i].index() != k {
                pattern_workspace.push_left(I::from_usize(i));
                flag_workspace[i] = I::from_usize(k);
                i = parents.get_parent(i).expect("enforced by ldl_symbolic");
            }
            pattern_workspace.push_left_on_right();
        }

        // use a sparse triangular solve to compute the values
        // of the kth row of L
        diag[k] = y_workspace[k];
        y_workspace[k] = N::zero();
        'pattern: for &i in pattern_workspace.iter_right() {
            let i = i.index();
            let yi = y_workspace[i];
            y_workspace[i] = N::zero();
            let p2 = l_colptr[i] + l_nz[i];
            for p in l_colptr[i].index()..p2.index() {
                // we cannot go inside this loop before something has actually
                // be written into l_indices[l_colptr[i]..p2] so this
                // read is actually not into garbage
                // actually each iteration of the 'pattern loop adds writes the
                // value in l_indices that will be read on the next iteration
                // TODO: can some design change make this fact more obvious?
                let y_index = l_indices[p].index();
                y_workspace[y_index] = y_workspace[y_index] - l_data[p] * yi;
            }
            let l_ki = yi / diag[i];
            diag[k] = diag[k] - l_ki * yi;
            l_indices[p2.index()] = I::from_usize(k);
            l_data[p2.index()] = l_ki;
            l_nz[i] += I::one();
        }
        if diag[k] == N::zero() {
            // FIXME should return info on k
            // but this would need breaking change in sprs error type
            return Err(SprsError::SingularMatrix);
        }
    }
    Ok(())
}

/// Triangular solve specialized on lower triangular matrices
/// produced by ldlt (diagonal terms are omitted and assumed to be 1).
pub fn ldl_lsolve<N, I, V: ?Sized>(l: &CsMatViewI<N, I>, x: &mut V)
where
    N: Clone + Copy + Num,
    I: SpIndex,
    V: IndexMut<usize, Output = N>,
{
    for (col_ind, vec) in l.outer_iterator().enumerate() {
        let x_col = x[col_ind];
        for (row_ind, &value) in vec.iter() {
            x[row_ind] = x[row_ind] - value * x_col;
        }
    }
}

/// Triangular transposed solve specialized on lower triangular matrices
/// produced by ldlt (diagonal terms are omitted and assumed to be 1).
pub fn ldl_ltsolve<N, I, V: ?Sized>(l: &CsMatViewI<N, I>, x: &mut V)
where
    N: Clone + Copy + Num,
    I: SpIndex,
    V: IndexMut<usize, Output = N>,
{
    for (outer_ind, vec) in l.outer_iterator().enumerate().rev() {
        let mut x_outer = x[outer_ind];
        for (inner_ind, &value) in vec.iter() {
            x_outer = x_outer - value * x[inner_ind];
        }
        x[outer_ind] = x_outer;
    }
}

#[cfg(test)]
mod test {
    use super::SymmetryCheck;
    use sprs::stack::DStack;
    use sprs::{self, linalg, CsMat, CsMatView, Permutation};

    fn test_mat1() -> CsMat<f64> {
        let indptr = vec![0, 2, 5, 6, 7, 13, 14, 17, 20, 24, 28];
        let indices = vec![
            0, 8, 1, 4, 9, 2, 3, 1, 4, 6, 7, 8, 9, 5, 4, 6, 9, 4, 7, 8, 0, 4,
            7, 8, 1, 4, 6, 9,
        ];
        let data = vec![
            1.7, 0.13, 1., 0.02, 0.01, 1.5, 1.1, 0.02, 2.6, 0.16, 0.09, 0.52,
            0.53, 1.2, 0.16, 1.3, 0.56, 0.09, 1.6, 0.11, 0.13, 0.52, 0.11, 1.4,
            0.01, 0.53, 0.56, 3.1,
        ];
        CsMat::new_csc((10, 10), indptr, indices, data)
    }

    fn test_vec1() -> Vec<f64> {
        vec![
            0.287, 0.22, 0.45, 0.44, 2.486, 0.72, 1.55, 1.424, 1.621, 3.759,
        ]
    }

    fn expected_factors1() -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>) {
        let expected_lp = vec![0, 1, 3, 3, 3, 7, 7, 10, 12, 13, 13];
        let expected_li = vec![8, 4, 9, 6, 7, 8, 9, 7, 8, 9, 8, 9, 9];
        let expected_lx = vec![
            0.076470588235294124,
            0.02,
            0.01,
            0.061547930450838589,
            0.034620710878596701,
            0.20003077396522542,
            0.20380058470533929,
            -0.0042935346524025902,
            -0.024807089102770519,
            0.40878266366119237,
            0.05752526570865537,
            -0.010068305077340346,
            -0.071852278207562709,
        ];
        let expected_d = vec![
            1.7,
            1.,
            1.5,
            1.1000000000000001,
            2.5996000000000001,
            1.2,
            1.290152331127866,
            1.5968603527854308,
            1.2799646117414738,
            2.7695677698030283,
        ];
        (expected_lp, expected_li, expected_lx, expected_d)
    }

    fn expected_lsolve_res1() -> Vec<f64> {
        vec![
            0.28699999999999998,
            0.22,
            0.45000000000000001,
            0.44,
            2.4816000000000003,
            0.71999999999999997,
            1.3972626557931991,
            1.3440844395148306,
            1.0599997771886431,
            2.7695677698030279,
        ]
    }

    fn expected_dsolve_res1() -> Vec<f64> {
        vec![
            0.16882352941176471,
            0.22,
            0.29999999999999999,
            0.39999999999999997,
            0.95460840129250657,
            0.59999999999999998,
            1.0830214557467768,
            0.84170443406044937,
            0.82814772179243734,
            0.99999999999999989,
        ]
    }

    fn expected_res1() -> Vec<f64> {
        vec![
            0.099999999999999992,
            0.19999999999999998,
            0.29999999999999999,
            0.39999999999999997,
            0.5,
            0.59999999999999998,
            0.70000000000000007,
            0.79999999999999993,
            0.90000000000000002,
            0.99999999999999989,
        ]
    }

    #[test]
    fn test_factor1() {
        let mut l_colptr = [0; 11];
        let mut parents = linalg::etree::ParentsOwned::new(10);
        let mut l_nz = [0; 10];
        let mut flag_workspace = [0; 10];
        let perm: Permutation<usize, &[usize]> = Permutation::identity(10);
        let mat = test_mat1();
        super::ldl_symbolic(
            mat.view(),
            &perm,
            &mut l_colptr,
            parents.view_mut(),
            &mut l_nz,
            &mut flag_workspace,
            SymmetryCheck::CheckSymmetry,
        );

        let nnz = l_colptr[10];
        let mut l_indices = vec![0; nnz];
        let mut l_data = vec![0.; nnz];
        let mut diag = [0.; 10];
        let mut y_workspace = [0.; 10];
        let mut pattern_workspace = DStack::with_capacity(10);
        super::ldl_numeric(
            mat.view(),
            &l_colptr,
            parents.view(),
            &perm,
            &mut l_nz,
            &mut l_indices,
            &mut l_data,
            &mut diag,
            &mut y_workspace,
            &mut pattern_workspace,
            &mut flag_workspace,
        )
        .unwrap();

        let (expected_lp, expected_li, expected_lx, expected_d) =
            expected_factors1();

        assert_eq!(&l_colptr, &expected_lp[..]);
        assert_eq!(&l_indices, &expected_li);
        assert_eq!(&l_data, &expected_lx);
        assert_eq!(&diag, &expected_d[..]);
    }

    #[test]
    fn test_solve1() {
        let (expected_lp, expected_li, expected_lx, expected_d) =
            expected_factors1();
        let b = test_vec1();
        let mut x = b.clone();
        let n = b.len();
        let l = CsMatView::new_view(
            sprs::CSC,
            (n, n),
            &expected_lp,
            &expected_li,
            &expected_lx,
        )
        .unwrap();
        super::ldl_lsolve(&l, &mut x);
        assert_eq!(&x, &expected_lsolve_res1());
        linalg::diag_solve(&expected_d, &mut x);
        assert_eq!(&x, &expected_dsolve_res1());
        super::ldl_ltsolve(&l, &mut x);

        let x0 = expected_res1();
        assert_eq!(x, x0);
    }

    #[test]
    fn test_factor_solve1() {
        let mat = test_mat1();
        let b = test_vec1();
        let ldlt = super::LdlNumeric::new(mat.view()).unwrap();
        let x = ldlt.solve(&b);
        let x0 = expected_res1();
        assert_eq!(x, x0);
    }

    #[test]
    fn permuted_ldl_solve() {
        // |1      | |1      | |1     2|   |1      | |1      2| |1      |
        // |  1    | |  2    | |  1 3  |   |    1  | |  21 6  | |    1  |
        // |  3 1  | |    3  | |    1  | = |  1    | |   6 2  | |  1    |
        // |2     1| |      4| |      1|   |      1| |2      8| |      1|
        //     L         D        L^T    =     P          A        P^T
        //
        // |1      2| |1|   | 9|
        // |  21 6  | |2|   |60|
        // |   6 2  | |3| = |18|
        // |2      8| |4|   |34|

        let mat = CsMat::new_csc(
            (4, 4),
            vec![0, 2, 4, 6, 8],
            vec![0, 3, 1, 2, 1, 2, 0, 3],
            vec![1, 2, 21, 6, 6, 2, 2, 8],
        );

        let perm = Permutation::new(vec![0, 2, 1, 3]);

        let ldlt = super::LdlNumeric::new_perm(mat.view(), perm).unwrap();
        let b = vec![9, 60, 18, 34];
        let x0 = vec![1, 2, 3, 4];
        let x = ldlt.solve(&b);
        assert_eq!(x, x0);
    }
}
