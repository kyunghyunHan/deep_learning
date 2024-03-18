use ndarray::prelude::*;
use rayon::prelude::*;

/*경사하강법 */
pub fn gradient_descent<F>(f: F, init_x: ArrayD<f64>, ir: f64, step_num: i32) -> ArrayD<f64>
where
    F: Fn() -> f64,
{
    let mut x = init_x.into_dyn();
    for _ in 0..step_num {
        let grad = numerical_gradient(&f, &mut x);
        x =x -(ir * &grad);
    }
    x.into_dyn()
}

pub fn numerical_gradient<F>(f: F, x: &mut ArrayD<f64>) -> ArrayD<f64>
where
    F: Fn() -> f64,
{
    let h = 1e-4; // 0.0001

    let mut grad = Array::zeros(x.raw_dim()); // 기울기 배열을 동적으로 할당

    for (x_val, o) in x.iter_mut().zip(grad.iter_mut()) {
        let tmp = *x_val; // copy
        *x_val = tmp + h;
        let y1 = f();
        *x_val = tmp - h;
        let y0 = f();
        *o = (y1 - y0) / (2. * h);
        *x_val = tmp;
    }

    grad
}
// pub fn numerical_gradient<F>(f: F, x: &mut ArrayD<f64>) -> ArrayD<f64>
// where
//     F: Fn() -> f64,
// {
//     let h = 1e-4; // 0.0001

//     let tmp = x.clone(); // x를 복제하여 임시 변수에 저장

//     // f(x + h) 계산
//     let mut x_plus_h = tmp.clone();
//     x_plus_h += h;
//     let y1 = f();

//     // f(x - h) 계산
//     let mut x_minus_h = tmp.clone();
//     x_minus_h -= h;
//     let y0 = f();

//     // 수치적 기울기 계산
//     let grad = (y1 - y0) / (2. * h);

//     // 모든 요소에 동일한 기울기 값 적용
//     let mut grad_vec = Array::from_elem(x.raw_dim(), grad);

//     grad_vec
// }