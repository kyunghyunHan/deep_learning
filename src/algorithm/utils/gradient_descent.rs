use ndarray::prelude::*;

/*경사하강법 */
pub fn gradient_descent<F>(f: F, init_x: ArrayD<f64>, ir: f64, step_num: i32) -> ArrayD<f64>
where
    F: Fn(ArrayD<f64>) -> f64,
{
    let mut x = init_x;
    for _ in 0..step_num {
        let grad = numerical_gradient(&f, x.clone().into_dyn());
        x = x - ir * grad;
    }
    x.into_dyn()
}



pub fn numerical_gradient<F>(f: F, x: ArrayD<f64>) -> ArrayD<f64>
where
    F: Fn(ArrayD<f64>) -> f64,
{
    let rank = x.ndim(); //rank설정
    let h = 1e-4;//0.0001

    match rank {
        1 => {
            let mut x = x.clone().into_dimensionality::<Ix1>().unwrap();
            let mut grad: Array1<f64> = Array::zeros(x.len()); //x와 형상이 같은 배열을 생성
            for idx in 0..x.len() {
                let tmp_val = x[idx];
                //f(x+h)계산
                x[idx] = tmp_val + h;
                let fxh1 = f(x.clone().into_dyn());

                //f(x-h)계산
                x[idx] = tmp_val - h;
                let fxh2 = f(x.clone().into_dyn());

                grad[idx] = (fxh1 - fxh2) / (2.0 * h);
                x[idx] = tmp_val;
            }

            return grad.into_dyn();
        }
        2 => {
            let x = x.clone().into_dimensionality::<Ix2>().unwrap();
            let mut grad = Array2::zeros(x.raw_dim());
            for (idx, mut row) in x.clone().axis_iter_mut(Axis(0)).enumerate() {
                for (j, val) in row.iter_mut().enumerate() {
                    let tmp_val = *val;
                    *val = tmp_val + h;
                    let fxh1 = f(x.clone().into_dyn());
                    *val = tmp_val - h;
                    let fxh2 = f(x.clone().into_dyn());
                    grad[[idx, j]] = (fxh1 - fxh2) / (2.0 * h);
                    *val = tmp_val;
                }
            }
            return grad.into_dyn();
        }
        _ => {
            panic!("error")
        }
    }
}
