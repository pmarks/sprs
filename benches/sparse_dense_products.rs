#[macro_use]
extern crate criterion;
use criterion::Criterion;

extern crate ndarray;
extern crate sprs;
extern crate flate2;

use ndarray::{Array, Array2, ShapeBuilder};
use sprs::{CsMat, CsVec};

fn sparse_dense_dotprod_default(c: &mut Criterion) {
    c.bench_function("sparse_dense_dotprod_default", |b| {

        let w = Array::range(0., 10., 0.00001);
        let x = CsVec::new(1000000, vec![0, 200000, 800000], vec![1., 2., 3.]);
        
        b.iter(|| { x.dot(&w); });
    });
}

fn sparse_dense_dotprod_specialized(c: &mut Criterion) {

    c.bench_function("sparse_dense_dotprod_specialized", |b| {
        let w = Array::range(0., 10., 0.00001);
        let x = CsVec::new(1000000, vec![0, 200000, 800000], vec![1., 2., 3.]);
        b.iter(|| { x.dot_dense(w.view()); });
    });
}

fn sparse_dense_vec_matprod_default(c: &mut Criterion) {

    c.bench_function("sparse_dense_vec_matprod_default", |b| {

        let w = Array::range(0., 10., 0.00001);
        let a = CsMat::new(
            (3, 1000000),
            vec![0, 2, 4, 5],
            vec![0, 1, 0, 2, 2],
            vec![1., 2., 3., 4., 5.],
        );

        b.iter(|| &a * &w );
    });
}

fn sparse_dense_vec_matprod_big(c: &mut Criterion) {

    let path = "/Users/patrick/code/sprs/filtered_feature_bc_matrix/matrix.mtx.gz";
    let f = std::fs::File::open(path).unwrap();
    let gz = flate2::read::GzDecoder::new(f);
    let mut reader = std::io::BufReader::new(gz);

    let tri_mat = sprs::io::read_matrix_market_from_bufread::<f32, usize, _>(&mut reader).unwrap();
    let _mat = tri_mat.to_csr();
    let mat = std::sync::Arc::new(_mat);
    let mymat = mat.clone();

    c.bench_function("sparse_dense_vec_matprod_big", move |b| {

        use ndarray::linalg::Dot;
        let cols = mat.as_ref().shape().1;
        let w = Array2::from_shape_fn((cols, 32), |(r,c)| (r+c) as f32);

        b.iter(|| { &mat.as_ref().dot(&w); });
    });
}



fn sparse_dense_vec_matprod_specialized(c: &mut Criterion) {
    c.bench_function("sparse_dense_vec_matprod_specialized", |b| {

        let w = Array::range(0., 10., 0.00001);
        let a = CsMat::new(
            (3, 1000000),
            vec![0, 2, 4, 5],
            vec![0, 1, 0, 2, 2],
            vec![1., 2., 3., 4., 5.],
        );
        let rows = a.rows();
        let cols = w.shape()[0];
        let w_reshape = w.view().into_shape((1, cols)).unwrap();
        let w_t = w_reshape.t();
        let mut res = Array2::zeros((rows, 1).f());

        b.iter(|| {
            sprs::prod::csr_mulacc_dense_colmaj(
                a.view(),
                w_t.view(),
                res.view_mut(),
            );
        })
    });
}

criterion_group!(
    benches,
    sparse_dense_dotprod_default,
    sparse_dense_dotprod_specialized,
    sparse_dense_vec_matprod_specialized,
    sparse_dense_vec_matprod_default,
);

criterion_group!{
    name = big;
    config = Criterion::default().sample_size(4).nresamples(5);
    targets = sparse_dense_vec_matprod_big,
}
criterion_main!(benches, big);
