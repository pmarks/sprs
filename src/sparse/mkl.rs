use std::ops::Deref;
use std::any::TypeId;

use indexing::SpIndex;
use ndarray::{ArrayView, ArrayViewMut, Axis};
use num_traits::Num;
use sparse::compressed::SpMatView;
use sparse::csmat::CompressedStorage;
///! Sparse matrix product
use sparse::prelude::*;
use std::iter::Sum;
use Ix2;

use sparse::spblas::{self, 
    sparse_operation_t, sparse_matrix_t, matrix_descr, 
    sparse_layout_t, sparse_status_t, 
    mkl_sparse_d_create_csr, mkl_sparse_d_create_csc,
    mkl_sparse_s_create_csr, mkl_sparse_s_create_csc, 
    mkl_sparse_d_mm,
    mkl_sparse_s_mm};


/// Wrapper around a MKL sparse matrix
pub struct MklSparseMatrix<'a, N>
{
    handle: sparse_matrix_t,
    phantom: &'a std::marker::PhantomData<N>,
    rows: usize,
    cols: usize,
    storage: CompressedStorage,
}

impl<'a, N> MklSparseMatrix<'a, N> {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

impl<'a, N> Drop for MklSparseMatrix<'a, N> {
    fn drop(&mut self) {
        unsafe { spblas::mkl_sparse_destroy(self.handle); }
    }
}


pub trait ToMklSparse<N> {
    fn to_mkl_sparse<'a>(&'a self) -> MklSparseMatrix<'a, N>;
}

impl<'a, I, IpS, IS, DS> ToMklSparse<f64> for CsMatBase<f64, I, IpS, IS, DS> where
    I: 'a + SpIndex + 'static,
    IpS: 'a + Deref<Target = [I]>,
    IS: 'a + Deref<Target = [I]>,
    DS: 'a + Deref<Target = [f64]>,
 {

    fn to_mkl_sparse<'b>(&'b self) -> MklSparseMatrix<'b, f64> {

        let mat = new_mkl_sparse_f64_i64_ptrs(
            self.storage,
            self.indptr.as_ptr() as *mut _,
            self.indices.as_ptr() as *mut _,
            self.data.as_ptr() as *mut _,
            self.rows() as i64,
            self.cols() as i64);

        MklSparseMatrix {
            handle: mat,
            phantom: &std::marker::PhantomData,
            rows: self.rows(),
            cols: self.cols(),
            storage: self.storage,
        }
    }
}

impl<'a, I, IpS, IS, DS> ToMklSparse<f32> for CsMatBase<f32, I, IpS, IS, DS> where
    I: 'a + SpIndex + 'static,
    IpS: 'a + Deref<Target = [I]>,
    IS: 'a + Deref<Target = [I]>,
    DS: 'a + Deref<Target = [f32]>,
 {

    fn to_mkl_sparse<'b>(&'b self) -> MklSparseMatrix<'b, f32> {

        let mat = new_mkl_sparse_f32_i64_ptrs(
            self.storage,
            self.indptr.as_ptr() as *mut _,
            self.indices.as_ptr() as *mut _,
            self.data.as_ptr() as *mut _,
            self.rows() as i64,
            self.cols() as i64);

        MklSparseMatrix {
            handle: mat,
            phantom: &std::marker::PhantomData,
            rows: self.rows(),
            cols: self.cols(),
            storage: self.storage,
        }
    }
}


fn new_mkl_sparse_f64_i64_ptrs(
    lhs_storage: CompressedStorage,
    index_ptr: *mut i64,
    indices: *mut i64,
    value_ptr: *mut f64,
    lhs_rows: i64,
    lhs_cols: i64,
) -> sparse_matrix_t
{

    // lhs sparse matrix in mkl internal format
    let mut mkl_mat: sparse_matrix_t = std::ptr::null_mut();

    let status = 
        if lhs_storage == CompressedStorage::CSR {
            unsafe {
                mkl_sparse_d_create_csr(&mut mkl_mat as *mut _,
                                                0,  // indexing is C-style / 0-based
                                                lhs_rows,
                                                lhs_cols,
                                                index_ptr, // rows start
                                                index_ptr.offset(1), // rows end
                                                indices,
                                                value_ptr,
                                                )
            } 
        } else {
            unsafe {
                mkl_sparse_d_create_csc(&mut mkl_mat as *mut _,
                                                0,  // indexing is C-style / 0-based
                                                lhs_rows,
                                                lhs_cols,
                                                index_ptr, // col start
                                                index_ptr.offset(1), // col end
                                                indices,
                                                value_ptr,
                                                )
            } 
        };

    if status != 0 {
        println!("got error in create csr: {}", status);
    }

    mkl_mat
}


fn new_mkl_sparse_f32_i64_ptrs(
    lhs_storage: CompressedStorage,
    index_ptr: *mut i64,
    indices: *mut i64,
    value_ptr: *mut f32,
    lhs_rows: i64,
    lhs_cols: i64,
) -> sparse_matrix_t
{

    // lhs sparse matrix in mkl internal format
    let mut mkl_mat: sparse_matrix_t = std::ptr::null_mut();

    let status = 
        if lhs_storage == CompressedStorage::CSR {
            unsafe {
                mkl_sparse_s_create_csr(&mut mkl_mat as *mut _,
                                                0,  // indexing is C-style / 0-based
                                                lhs_rows,
                                                lhs_cols,
                                                index_ptr, // rows start
                                                index_ptr.offset(1), // rows end
                                                indices,
                                                value_ptr,
                                                )
            } 
        } else {
            unsafe {
                mkl_sparse_s_create_csc(&mut mkl_mat as *mut _,
                                                0,  // indexing is C-style / 0-based
                                                lhs_rows,
                                                lhs_cols,
                                                index_ptr, // col start
                                                index_ptr.offset(1), // col end
                                                indices,
                                                value_ptr,
                                                )
            } 
        };

    if status != 0 {
        println!("got error in create csr: {}", status);
    }

    mkl_mat
}


/// Sparse-dense matrix multiply with MKL, f64 values
pub fn sparse_mulacc_dense_rowmaj_f64_i64_ptrs(
    lhs: &MklSparseMatrix<f64>,
    rhs_ptr: *mut f64,
    rhs_cols: i64,
    out_ptr: *mut f64,
    out_cols: i64,
)
{

    let op = spblas::sparse_operation_t_SPARSE_OPERATION_NON_TRANSPOSE;

    let descr = matrix_descr {
        type_: spblas::sparse_matrix_type_t_SPARSE_MATRIX_TYPE_GENERAL,
        mode: spblas::sparse_fill_mode_t_SPARSE_FILL_MODE_FULL,
        diag: 0,
    };

    // layout of of dense matrices
    let layout = spblas::sparse_layout_t_SPARSE_LAYOUT_ROW_MAJOR;

    // The mkl_sparse_?_mm routine performs a matrix-matrix operation:
    // y := alpha*op(A)*x + beta*y
    // where alpha and beta are scalars, A is a sparse matrix, and x and y are dense matrices.
    let status = unsafe { 
        mkl_sparse_d_mm(
            op,
            1.0,
            lhs.handle,
            descr,
            layout,
            rhs_ptr,  // dense matrix: *const f32,
            out_cols, // columns in y / out
            rhs_cols, // leading dimension of matrix x, in the in-memory layout
            1.0,
            out_ptr,
            out_cols, // leading dimension of matrix y, in the in-memory layout
    ) };

    if status != 0 {
        panic!("got error in create mkl_sparse_d_mm: {}", status);
    }
}


/// Sparse-dense matrix multiply with MKL, f32 values
pub fn sparse_mulacc_dense_rowmaj_f32_i64_ptrs(
    lhs: &MklSparseMatrix<f32>,
    rhs_ptr: *mut f32,
    rhs_cols: i64,
    out_ptr: *mut f32,
    out_cols: i64,
)
{

    let op = spblas::sparse_operation_t_SPARSE_OPERATION_NON_TRANSPOSE;

    let descr = matrix_descr {
        type_: spblas::sparse_matrix_type_t_SPARSE_MATRIX_TYPE_GENERAL,
        mode: spblas::sparse_fill_mode_t_SPARSE_FILL_MODE_FULL,
        diag: 0,
    };

    // layout of of dense matrices
    let layout = spblas::sparse_layout_t_SPARSE_LAYOUT_ROW_MAJOR;

    // The mkl_sparse_?_mm routine performs a matrix-matrix operation:
    // y := alpha*op(A)*x + beta*y
    // where alpha and beta are scalars, A is a sparse matrix, and x and y are dense matrices.
    let status = unsafe { 
        mkl_sparse_s_mm(
            op,
            1.0,
            lhs.handle,
            descr,
            layout,
            rhs_ptr,  // dense matrix: *const f32,
            out_cols, // columns in y / out
            rhs_cols, // leading dimension of matrix x, in the in-memory layout
            1.0,
            out_ptr,
            out_cols, // leading dimension of matrix y, in the in-memory layout
    ) };

    if status != 0 {
        panic!("got error in create mkl_sparse_s_mm: {}", status);
    }
}


use ndarray::{self, Array, ArrayBase, linalg::Dot};

#[inline(always)]
/// Return `true` if `A` and `B` are the same type
fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

impl<'a, 'b, DS> Dot<ArrayBase<DS, Ix2>> for MklSparseMatrix<'a, f64>
where
    DS: 'b + ndarray::Data<Elem = f64>,
{
    type Output = Array<f64, Ix2>;

    fn dot(&self, rhs: &ArrayBase<DS, Ix2>) -> Array<f64, Ix2> {

        let rows = self.rows();
        let cols = rhs.shape()[1];
        let mut out = Array::zeros((rows, cols));

        sparse_mulacc_dense_rowmaj_f64_i64_ptrs(
            self,
            rhs.as_ptr() as *mut _,
            rhs.shape()[1] as i64,
            out.as_mut_ptr() as *mut _,
            out.shape()[1] as i64,
        );

        out
    }
}


impl<'a, 'b, DS> Dot<ArrayBase<DS, Ix2>> for MklSparseMatrix<'a, f32>
where
    DS: 'b + ndarray::Data<Elem = f64>,
{
    type Output = Array<f32, Ix2>;

    fn dot(&self, rhs: &ArrayBase<DS, Ix2>) -> Array<f32, Ix2> {

        let rows = self.rows();
        let cols = rhs.shape()[1];
        let mut out = Array::zeros((rows, cols));

        sparse_mulacc_dense_rowmaj_f32_i64_ptrs(
            self,
            rhs.as_ptr() as *mut _,
            rhs.shape()[1] as i64,
            out.as_mut_ptr() as *mut _,
            out.shape()[1] as i64,
        );

        out
    }
}