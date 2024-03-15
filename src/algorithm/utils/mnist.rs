use ndarray::prelude::*;
use polars::prelude::*;
pub struct Mnist {
    pub x_train: Array2<f64>,
    pub y_train: Array1<f64>,
    pub x_test: Array2<f64>,
    pub y_test: Array1<f64>,
}
impl Mnist {
    pub fn load_mnist() -> Mnist {
        let train_df = CsvReader::from_path("./datasets/mnist/train.csv")
            .unwrap()
            .finish()
            .unwrap();
        let test_df = CsvReader::from_path("./datasets/mnist/test.csv")
            .unwrap()
            .finish()
            .unwrap();
        let submission_df = CsvReader::from_path("./datasets/mnist/sample_submission.csv")
            .unwrap()
            .finish()
            .unwrap();

        let y_train = arr1(
            &train_df
                .column("label")
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<f64>>(),
        );

        let y_test = arr1(
            &submission_df
                .column("Label")
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<f64>>(),
        );

        let x_train = train_df
            .drop("label")
            .unwrap()
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)
            .unwrap();
        let x_test = test_df
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

pub fn load_mnist() -> Mnist {
    Mnist::load_mnist()
}
