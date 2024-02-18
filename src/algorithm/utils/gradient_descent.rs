use ndarray::prelude::*;

/*경사하강법 */
// pub fn gradient_descent<F>(f: F, init_x: ArrayD<f64>, ir: f64, step_num: i32) -> ArrayD<f64>
// where
//     F: Fn(&ArrayD<f64>) -> f64,
// {
//     let mut x = init_x;
//     for _ in 0..step_num {
//         let grad = numerical_gradient(&f, &mut x.into_dyn());
//         x = x - ir * grad;
//     }
//     x.into_dyn()
// }



pub fn numerical_gradient<F>(f: F, x: &mut ArrayD<f64>) -> ArrayD<f64>
where
    F: Fn() -> f64,
{
    let h = 1e-4;//0.0001
   
    let mut x = x.clone().into_dimensionality::<Ix2>().unwrap();
    let mut grad: Array2<f64> = Array::zeros(x.raw_dim()); //x와 형상이 같은 배열을 생성
    for (x, o) in x.iter_mut().zip(grad.iter_mut()) {
        let tmp = *x; // copy
        *x = tmp + h;
        let y1 = f();
        *x = tmp - h;
        let y0 = f();
        *o = (y1 - y0) / (2.*h);
        *x = tmp;
    }
    grad.into_dyn()
}
