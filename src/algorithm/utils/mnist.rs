use ndarray::prelude::*;
use polars::prelude::*;
pub struct Mnist {
   pub  x_train: Array2<f64>,
   pub y_train: Array2<f64>,
   pub  x_test: Array2<f64>,
   pub  y_test: Array2<f64>,
}
impl Mnist {
   pub  fn load_mnist() -> Mnist {
        let x_train = CsvReader::from_path("./dataset/mnist/x_train.csv")
            .unwrap()
            .has_header(false)
            .finish()
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();
        let y_train = CsvReader::from_path("./dataset/mnist/y_train.csv")
            .unwrap()
            .has_header(false)
            .finish()
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();
        let x_test = CsvReader::from_path("./dataset/mnist/x_test.csv")
            .unwrap()
            .has_header(false)
            .finish()
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();
        let y_test = CsvReader::from_path("./dataset/mnist/t_test.csv")
            .unwrap()
            .has_header(false)
            .finish()
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();

        Mnist {
            x_train,
            y_train,
            x_test,
            y_test,
        }
    }
}

pub fn load_mnist()->Mnist{
  Mnist::load_mnist()
}