//! The SSE3, SSE4.1, AVX and FMA intrinsics rustfft 6.4's SSE and AVX paths call:
//! moveldup / movehdup, addsub, blend, permute, broadcast, the 128-to-256 casts, insertf128,
//! fmaddsub / fmsubadd and loadu2_m128d. Under rustc these check the expectations against
//! stdarch; under Relax they check its std against the same numbers.
#![cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn lanes4(v: __m128) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    unsafe { _mm_storeu_ps(out.as_mut_ptr(), v) };
    out
}

fn lanes8(v: __m256) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(out.as_mut_ptr(), v) };
    out
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse() -> [[f32; 4]; 4] {
    let a = _mm_setr_ps(0.0, 1.0, 2.0, 3.0);
    let b = _mm_setr_ps(10.0, 20.0, 30.0, 40.0);
    [
        lanes4(_mm_moveldup_ps(a)),
        lanes4(_mm_addsub_ps(a, b)),
        lanes4(_mm_blend_ps(a, b, 0b0101)),
        lanes4(_mm_movehdup_ps(a)),
    ]
}

#[test]
fn sse3_and_sse41_forms() {
    if !std::arch::is_x86_feature_detected!("sse4.1") {
        return;
    }
    let [ldup, addsub, blend, hdup] = unsafe { sse() };
    assert_eq!(ldup, [0.0, 0.0, 2.0, 2.0], "even lanes twice");
    assert_eq!(hdup, [1.0, 1.0, 3.0, 3.0], "odd lanes twice");
    assert_eq!(addsub, [-10.0, 21.0, -28.0, 43.0], "even a - b, odd a + b");
    assert_eq!(blend, [10.0, 1.0, 30.0, 3.0], "b where the bit is set");
}

#[target_feature(enable = "avx")]
unsafe fn avx() -> ([f32; 4], [f32; 4], [[f32; 8]; 7], [f64; 4]) {
    let a4 = _mm_setr_ps(0.0, 1.0, 2.0, 3.0);
    let a = _mm256_setr_ps(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
    let b = _mm256_setr_ps(10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0);
    let five = 5.0f32;
    let lo = [1.0f64, 2.0];
    let hi = [3.0f64, 4.0];
    let mut pd = [0.0f64; 4];
    _mm256_storeu_pd(pd.as_mut_ptr(), _mm256_loadu2_m128d(hi.as_ptr(), lo.as_ptr()));
    (
        lanes4(_mm_permute_ps(a4, 0b00_01_10_11)),
        lanes4(_mm_broadcast_ss(&five)),
        [
            lanes8(_mm256_moveldup_ps(a)),
            lanes8(_mm256_movehdup_ps(a)),
            lanes8(_mm256_blend_ps(a, b, 0b1000_0001)),
            lanes8(_mm256_permute_ps(a, 0b00_01_10_11)),
            lanes8(_mm256_broadcast_ss(&five)),
            lanes8(_mm256_zextps128_ps256(a4)),
            lanes8(_mm256_insertf128_ps(a, _mm_setr_ps(9.0, 9.5, 8.0, 8.5), 1)),
        ],
        pd,
    )
}

#[test]
fn avx_forms() {
    if !std::arch::is_x86_feature_detected!("avx") {
        return;
    }
    let (permute4, broadcast4, out, pd) = unsafe { avx() };
    assert_eq!(permute4, [3.0, 2.0, 1.0, 0.0], "two bits per lane pick a's lane");
    assert_eq!(broadcast4, [5.0; 4]);
    let [ldup, hdup, blend, permute, broadcast, zext, insert] = out;
    assert_eq!(ldup, [0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0]);
    assert_eq!(hdup, [1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0]);
    assert_eq!(blend, [10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 17.0], "bits 0 and 7 from b");
    assert_eq!(permute, [3.0, 2.0, 1.0, 0.0, 7.0, 6.0, 5.0, 4.0], "the same control on each half");
    assert_eq!(broadcast, [5.0; 8]);
    assert_eq!(zext, [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0], "the upper half zero");
    assert_eq!(insert, [0.0, 1.0, 2.0, 3.0, 9.0, 9.5, 8.0, 8.5], "the upper half replaced");
    assert_eq!(pd, [1.0, 2.0, 3.0, 4.0], "low half from loaddr, high from hiaddr");
}

#[target_feature(enable = "avx,fma")]
unsafe fn fma() -> ([f32; 4], [f32; 4], [f32; 8], [f32; 8]) {
    let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
    let b = _mm_set1_ps(2.0);
    let c = _mm_set1_ps(1.0);
    let a8 = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
    let b8 = _mm256_set1_ps(2.0);
    let c8 = _mm256_set1_ps(1.0);
    (
        lanes4(_mm_fmaddsub_ps(a, b, c)),
        lanes4(_mm_fmsubadd_ps(a, b, c)),
        lanes8(_mm256_fmaddsub_ps(a8, b8, c8)),
        lanes8(_mm256_fmsubadd_ps(a8, b8, c8)),
    )
}

#[test]
fn fma_alternating_forms() {
    if !(std::arch::is_x86_feature_detected!("avx") && std::arch::is_x86_feature_detected!("fma")) {
        return;
    }
    let (addsub, subadd, addsub8, subadd8) = unsafe { fma() };
    assert_eq!(addsub, [1.0, 5.0, 5.0, 9.0], "even a*b - c, odd a*b + c");
    assert_eq!(subadd, [3.0, 3.0, 7.0, 7.0], "even a*b + c, odd a*b - c");
    assert_eq!(addsub8, [1.0, 5.0, 5.0, 9.0, 9.0, 13.0, 13.0, 17.0]);
    assert_eq!(subadd8, [3.0, 3.0, 7.0, 7.0, 11.0, 11.0, 15.0, 15.0]);
}
