#[macro_use]
extern crate criterion;
use criterion::Criterion;

extern crate alga;
extern crate sprs;

use alga::general::{Additive, Inverse};
use sprs::CsVec;

fn csvec_neg(c: &mut Criterion) {
    c.bench_function("csvec_neg", |b| {
        let vector = CsVec::new(10000, (10..9000).collect::<Vec<_>>(), vec![-1.3; 8990]);
        b.iter(|| -vector.clone())
    });
}

fn csvec_additive_inverse(c: &mut Criterion) {
    
    c.bench_function("csvec_additive_inverse", |b| {
        let vector = CsVec::new(10000, (10..9000).collect::<Vec<_>>(), vec![-1.3; 8990]);
        b.iter(|| Inverse::<Additive>::inverse(&vector))
    });
}

criterion_group!(benches, csvec_neg, csvec_additive_inverse);
criterion_main!(benches);
