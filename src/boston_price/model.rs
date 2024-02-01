// use burn::nn::loss::MSELoss;
// use burn::{
//     backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu},
//     config::Config,
//     data::dataloader::{batcher::Batcher, DataLoaderBuilder},
//     module::Module,
//     nn::{
//         conv::{Conv2d, Conv2dConfig}, //기본convolution
//         loss::CrossEntropyLoss,
//         pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
//         Dropout,
//         DropoutConfig,
//         Linear,
//         LinearConfig,
//         ReLU,
//     },
//     optim::AdamConfig,
//     record::CompactRecorder,
//     tensor::{
//         backend::{AutodiffBackend, Backend},
//         Data, Float, Int, Tensor,
//     },
//     train::{
//         metric::{AccuracyMetric, LossMetric},
//         ClassificationOutput, LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
//     },
// };
// use rayon::prelude::*;
// use std::fs::File;
// extern crate serde;

// use burn::data::dataset::{Dataset, InMemDataset};
// use burn::record::Recorder;
// use burn::tensor::ElementConversion;
// use serde::{Deserialize, Deserializer, Serialize, Serializer};
// // use serde_bytes::ByteBuf;
// use burn::data::dataset::transform::{Mapper, MapperDataset};
// use burn::data::dataset::SqliteDataset;
// use burn::nn::loss::Reduction;
// use image::GenericImageView;
// use polars::prelude::*;
// use serde::de::Error;
// use std::fmt;
// use std::marker::PhantomData;
// use std::path::Path;
// // use std::fs::File;
// #[derive(Deserialize, Serialize, Debug, Clone)]
// struct BostonData {
//     #[serde(rename = "CRIM")]
//     pub crim: f64,
//     #[serde(rename = "ZN")]
//     pub zn: f64,
//     #[serde(rename = "INDUS")]
//     pub indus: f64,
//     #[serde(rename = "CHAS")]
//     pub chas: f64,
//     #[serde(rename = "NOX")]
//     pub nox: f64,
//     #[serde(rename = "RM")]
//     pub rm: f64,
//     #[serde(rename = "AGE")]
//     pub age: f64,
//     #[serde(rename = "DIS")]
//     pub dis: f64,
//     #[serde(rename = "RAD")]
//     pub rad: f64,
//     #[serde(rename = "TAX")]
//     pub tax: f64,
//     #[serde(rename = "PTRATIO")]
//     pub ptratio: f64,
//     #[serde(rename = "B")]
//     pub b: f64,
//     #[serde(rename = "LSTAT")]
//     pub lstat: f64,
//     #[serde(rename = "MEDV")]
//     pub medv: f64,
// }

// #[derive(Deserialize, Serialize, PartialEq, Debug, Clone)]
// struct BostonData2 {
//     pub house_data: [f64; 13],
//     pub medv: f64,
// }
// // #[derive(Debug)]
// pub struct Datasets {
//     dataset: Vec<BostonData2>,
// }
// #[derive(Clone, Debug)]
// pub struct Test<B: Backend> {
//     pub house_data: Tensor<B, 1, Float>,
//     pub medv: Tensor<B, 1, Float>,
// }
// impl Dataset<BostonData2> for Datasets {
//     fn get(&self, index: usize) -> Option<BostonData2> {
//         self.dataset.get(index).cloned()
//     }

//     fn len(&self) -> usize {
//         self.dataset.len()
//     }
// }
// impl Datasets {
//     pub fn new() -> Self {
//         let train_df = CsvReader::from_path("./dataset/housing_data/HousingData.csv")
//             .unwrap()
//             .finish()
//             .unwrap();
//         let labels: Vec<f64> = train_df
//             .column("MEDV")
//             .unwrap()
//             .f64()
//             .unwrap()
//             .into_no_null_iter()
//             .collect();

//         let house_data = train_df
//             .drop("MEDV")
//             .unwrap()
//             .to_ndarray::<Float64Type>(IndexOrder::Fortran)
//             .unwrap();

//         let mut x_train_vec: Vec<Vec<_>> = Vec::new();
//         for row in house_data.outer_iter() {
//             let row_vec: Vec<_> = row.iter().cloned().collect();
//             x_train_vec.push(row_vec);
//         }
//         // 3차원 배열로 만들어야 함
//         let mut boston_data_vec: Vec<BostonData2> = Vec::new();

//         for k in 0..labels.len() {
//             boston_data_vec.push(BostonData2 {
//                 medv: labels[k],
//                 house_data: x_train_vec[k].as_slice().try_into().unwrap(),
//             });
//         }
//         let data_sets = Datasets {
//             dataset: boston_data_vec,
//         };
//         data_sets
//     }
// }
// impl<B: Backend> Batcher<BostonData2, Test<B>> for Tester<B> {
//     fn batch(&self, items: Vec<BostonData2>) -> Test<B> {
//         let house_data = items
//             .iter()
//             .map(|item| Data::<f64, 1>::from(item.house_data))
//             .map(|data| Tensor::<B, 1>::from_data(data.convert()))
//             // .map(|tensor| tensor.reshape([1, 13]))
//             // // .map(|tensor: Tensor<B, 2>| ((tensor / 255) - 0.1307) / 0.3081)
//             .collect();

//         let medv = items
//             .iter()
//             .map(|item| Tensor::<B, 1, Float>::from_data(Data::from([(item.medv as i64).elem()])))
//             .collect();

//         let house_data = Tensor::cat(house_data, 0).to_device(&self.device);

//         let medv = Tensor::cat(medv, 0).to_device(&self.device);

//         Test { house_data, medv }
//     }
// }

// #[derive(Module, Debug)] //딥러닝 모듈생성
// pub struct Model<B: Backend> {
//     //BackEnd:새모델이 모든 벡엔드에서 실행할수 있게함
//     // conv1: Conv2d<B>,
//     // conv2: Conv2d<B>,
//     // pool: AdaptiveAvgPool2d,
//     dropout: Dropout,
//     linear1: Linear<B>,
//     linear2: Linear<B>,
//     activation: ReLU,
// }
// //network의 기본 구성
// //구성을 직렬화하여 모델 hyperprameter를 쉽게 저장
// #[derive(Config, Debug)]
// pub struct ModelConfig {
//     num_classes: usize, //
//     hidden_size: usize, //
//     #[config(default = "0.5")]
//     dropout: f64, //dropout
// }
// #[derive(Config)]
// pub struct TrainingConfig {
//     pub model: ModelConfig,
//     pub optimizer: AdamConfig,
//     #[config(default = 10)]
//     pub num_epochs: usize,
//     #[config(default = 64)]
//     pub batch_size: usize,
//     #[config(default = 4)]
//     pub num_workers: usize,
//     #[config(default = 42)]
//     pub seed: u64,
//     #[config(default = 1.0e-4)]
//     pub learning_rate: f64,
// }
// pub struct Tester<B: Backend> {
//     device: B::Device,
// }

// impl<B: Backend> Tester<B> {
//     pub fn new(device: B::Device) -> Self {
//         Self { device }
//     }
// }
// impl ModelConfig {
//     /// Returns the initialized model.
//     pub fn init<B: Backend>(&self) -> Model<B> {
//         Model {
//             //커널 크기 3사용
//             //채널 1에서 8로 확장
//             // conv1: Conv2dConfig::new([1, 8], [3, 3]).init(),
//             // // //8에서 16으로 확장
//             // conv2: Conv2dConfig::new([8, 16], [3, 3]).init(),
//             // //적응형 평균 폴링 모듈을 사용 이미지의 차원을 8x8으로 축소
//             // pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
//             activation: ReLU::new(),
//             linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(),
//             linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
//             dropout: DropoutConfig::new(self.dropout).init(),
//         }
//     }
//     pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
//         Model {
//             // conv1: Conv2dConfig::new([1, 8], [3, 3]).init_with(record.conv1),
//             // conv2: Conv2dConfig::new([8, 16], [3, 3]).init_with(record.conv2),
//             // pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
//             activation: ReLU::new(),
//             linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init_with(record.linear1),
//             linear2: LinearConfig::new(self.hidden_size, self.num_classes)
//                 .init_with(record.linear2),
//             dropout: DropoutConfig::new(self.dropout).init(),
//         }
//     }
// }
// // impl<B: AutodiffBackend> TrainStep<Test<B>, ClassificationOutput<B>> for Model<B> {
// //     fn step(&self, batch: Test<B>) -> TrainOutput<ClassificationOutput<B>> {
// //         let item = self.forward_regression(batch.house_data, batch.medv);

// //         TrainOutput::new(self, item.loss.backward(), item)
// //     }
// // }

// // impl<B: Backend> ValidStep<Test<B>, ClassificationOutput<B>> for Model<B> {
// //     fn step(&self, batch: Test<B>) -> ClassificationOutput<B> {
// //         self.forward_regression(batch.house_data, batch.medv)
// //     }
// // }
// // impl Dataset<BostonData2> for Datasets {
// //     // fn get(&self, index: usize) -> BostonData2 {
// //     //     self.dataset.get(index)
// //     // }
// //     fn len(&self) -> usize {
// //         self.dataset.len()
// //     }
// // }

// pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
//     std::fs::create_dir_all(artifact_dir).ok();
//     config
//         .save(format!("{artifact_dir}/config.json"))
//         .expect("Config should be saved successfully");

//     B::seed(config.seed);
//     let batcher_train = Tester::<B>::new(device.clone());
//     let batcher_valid = Tester::<B::InnerBackend>::new(device.clone());

//     let dataloader_train = DataLoaderBuilder::new(batcher_train)
//         .batch_size(config.batch_size)
//         .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(Datasets::new());

//     let dataloader_test = DataLoaderBuilder::new(batcher_valid)
//         .batch_size(config.batch_size)
//         .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(Datasets::new());

//     let learner = LearnerBuilder::new(artifact_dir)
//         .metric_train_numeric(AccuracyMetric::new())
//         .metric_valid_numeric(AccuracyMetric::new())
//         .metric_train_numeric(LossMetric::new())
//         .metric_valid_numeric(LossMetric::new())
//         .with_file_checkpointer(CompactRecorder::new())
//         .devices(vec![device])
//         .num_epochs(config.num_epochs)
//         .build(
//             config.model.init::<B>(),
//             config.optimizer.init(),
//             config.learning_rate,
//         );

//     let model_trained = learner.fit(dataloader_train, dataloader_test);
//     model_trained
//         .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
//         .expect("Trained model should be saved successfully");
// }
// /* Model method */
// impl<B: Backend> Model<B> {
//     pub fn forward(&self, images: Tensor<B, 1>) -> Tensor<B, 2> {
//         let [batch_size] = images.dims();

//         // Create a channel at the second dimension.
//         let x = images.reshape([batch_size, 1]);

//         // let x = self.conv1.forward(x); // [batch_size, 8, _, _]
//         let x = self.dropout.forward(x);
//         // let x = self.conv2.forward(x); // [batch_size, 16, _, _]
//         let x = self.dropout.forward(x);
//         let x = self.activation.forward(x);
//         /*채널, */
//         // let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
//         let x = x.reshape([batch_size, 1]);
//         let x = self.linear1.forward(x);
//         let x = self.dropout.forward(x);
//         let x = self.activation.forward(x);

//         /*1024(16 _ 8 _ 8) */
//         self.linear2.forward(x) // [batch_size, num_classes]
//     }

//     // pub fn forward_classification(
//     //     &self,
//     //     images: Tensor<B, 1>,
//     //     targets: Tensor<B, 1,Float>,
//     // ) -> RegressionOutput<B> {
//     //     let output = self.forward(images);
//     //     let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets.clone());

//     //     RegressionOutput::new(loss, output, targets)
//     // }
//     pub fn forward_regression(
//         &self,
//         images: Tensor<B, 1>,
//         targets: Tensor<B, 2, Float>,
//     ) -> RegressionOutput<B> {
//         let targets = targets;
//         let output = self.forward(images);
//         let loss = MSELoss::new();
//         let loss = loss.forward(output.clone(), targets.clone(), Reduction::Auto);

//         RegressionOutput {
//             loss,
//             output,
//             targets,
//         }
//     }
// }
// pub fn main() {
//     let train_df = CsvReader::from_path("./dataset/housing_data/HousingData.csv")
//         .unwrap()
//         .finish()
//         .unwrap();
//     println!(
//         "{}",
//         train_df
//             .to_ndarray::<Float64Type>(IndexOrder::Fortran)
//             .unwrap()
//     );
//     Datasets::new();
//     // println!("{}", Datasets::new().unwrap())
// }
