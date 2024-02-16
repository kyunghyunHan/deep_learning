use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

pub fn softmax(a: ArrayD<f64>) -> ArrayD<f64> {
    let rank = a.ndim();
    if rank == 1 {
        let a = a.clone().into_dimensionality::<Ix1>().unwrap();
        let c: f64 = a[a.argmax().unwrap()];
        let exp_a = a.mapv(|x| (x - c).exp());
        let sum_exp_a = exp_a.sum();
        (exp_a / sum_exp_a).into_dyn()
    } else if rank == 2 {
        let a = a.clone().into_dimensionality::<Ix2>().unwrap();
        let exp_a = a.mapv(f64::exp);
        let sum_exp_a = exp_a.sum_axis(Axis(1));
        exp_a / sum_exp_a.insert_axis(Axis(1)).into_dyn()
    } else {
        panic!("rank error")
    }
}

//시그모이드
pub fn sigmoid(x: ArrayD<f64>) -> ArrayD<f64> {     
    x.mapv(|element| 1.0 / (1.0 + (-element).exp())).into_dyn()
}

