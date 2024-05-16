// use burn::{
//     config::Config,
//     module::Module,
//     nn::{
//         conv::{Conv2d, Conv2dConfig},//기본convolution
//         pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
//         Dropout, DropoutConfig, Linear, LinearConfig, ReLU,
//         loss::CrossEntropyLoss
//     },
//     tensor::{
//         backend::{
//             Backend,AutodiffBackend
//         }, Data, Int, Tensor
//     },
//     train::{
//         ClassificationOutput,TrainStep,TrainOutput,ValidStep,LearnerBuilder,
//         metric::{
//             AccuracyMetric,
//             LossMetric
//         }
//     },
//     optim::AdamConfig,
//     data::dataloader::{DataLoaderBuilder,batcher::Batcher}, 
//     record::CompactRecorder,
//     backend::{
//         Autodiff,
//         Wgpu,
//         wgpu::AutoGraphicsApi
//     }
// };
// use serde::{Deserialize, Serialize};
// use burn::data::dataset::{Dataset, InMemDataset};
// use std::path::Path;
// use burn::tensor::ElementConversion;
// use burn::record::Recorder;


// /* 데이터 구조체 */
// #[derive(Deserialize, Serialize, Debug, Clone)]
// pub struct DiabetesPatient {
//     #[serde(rename = "Width")]

//     pub width:i64,

//     #[serde(rename = "Height")]
//     pub height:i64
// }
// /*데이터 셋 구조체 */
// pub struct DiabetesDataset {
//     dataset: InMemDataset<DiabetesPatient>,
// }

// /*Model
// - 모델설정
// - 기본 컨볼류션 신경망
// - 드롭아웃:훈련성능을 향상시키기 위해
// - linear
// - activarion

// */


// /*

// Tensor<B, 3> // Float tensor (default)
// Tensor<B, 3, Float> // Float tensor (explicit)
// Tensor<B, 3, Int> // Int tensor
// Tensor<B, 3, Bool> // Bool tensor

// 자세한 타입은 백엔드에서 결정

// */
// #[derive(Module, Debug)]//딥러닝 모듈생성
// pub struct Model<B: Backend> {//BackEnd:새모델이 모든 벡엔드에서 실행할수 있게함
//     // conv1: Conv2d<B>,
//     // conv2: Conv2d<B>,
//     pool: AdaptiveAvgPool2d,
//     dropout: Dropout,
//     linear1: Linear<B>,
//     linear2: Linear<B>,
//     activation: ReLU,
// }

// /* Model method */
// impl<B: Backend> Model<B> {
  
//     pub fn forward(&self, images: Tensor<B, 1>) -> Tensor<B, 2> {

        
//         let [batch_size] = images.dims();

//         // Create a channel at the second dimension.
//         let x = images.reshape([batch_size,  1]);


        
//         // let x = self.conv1.forward(x); // [batch_size, 8, _, _]
//         let x = self.dropout.forward(x);
//         // let x = self.conv2.forward(x); // [batch_size, 16, _, _]
//         let x = self.dropout.forward(x);
//         let x = self.activation.forward(x);
//         /*채널, */                 
//         // let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
//         let x = x.reshape([batch_size, 1 ]);
//         let x = self.linear1.forward(x);
//         let x = self.dropout.forward(x);
//         let x = self.activation.forward(x);
   
//         /*1024(16 _ 8 _ 8) */
//         self.linear2.forward(x) // [batch_size, num_classes]
//     }

//     pub fn forward_classification(
//         &self,
//         images: Tensor<B,1>,
//         targets: Tensor<B, 1, Int>,
//     ) -> ClassificationOutput<B> {
//         let output = self.forward(images);
//         let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets.clone());

//         ClassificationOutput::new(loss, output, targets)
//     }

// }
// //network의 기본 구성  
// //구성을 직렬화하여 모델 hyperprameter를 쉽게 저장
// #[derive(Config, Debug)]
// pub struct ModelConfig {
//     num_classes: usize,//
//     hidden_size: usize,//
//     #[config(default = "0.5")]
//     dropout: f64,//dropout
// }
// impl DiabetesDataset {
//     pub fn new() -> Result<Self, std::io::Error> {
//         // Download dataset csv file
//         let path = Path::new(file!()).parent().unwrap().parent().unwrap().join("../dataset/test.csv");

//         // Build dataset from csv with tab ('\t') delimiter
//         let mut rdr = csv::ReaderBuilder::new().delimiter(b'\t');

//         let dataset = InMemDataset::from_csv(path).unwrap();
//         println!("{:?}",dataset.get(0));
//         let dataset = Self { dataset };
//         Ok(dataset)
//     }
// }

// impl Dataset<DiabetesPatient> for DiabetesDataset {
//     fn get(&self, index: usize) -> Option<DiabetesPatient> {
//         self.dataset.get(index)
//     }
//     fn len(&self) -> usize {
//         self.dataset.len()
//     }
// }





// impl<B: AutodiffBackend> TrainStep<Test<B>, ClassificationOutput<B>> for Model<B> {
//     fn step(&self, batch: Test<B>) -> TrainOutput<ClassificationOutput<B>> {
//         let item = self.forward_classification(batch.widths, batch.targets);

//         TrainOutput::new(self, item.loss.backward(), item)
//     }
// }

// impl<B: Backend> ValidStep<Test<B>, ClassificationOutput<B>> for Model<B> {
//     fn step(&self, batch: Test<B>) -> ClassificationOutput<B> {
//         self.forward_classification(batch.widths, batch.targets)
//     }
// }


// impl ModelConfig {
//     /// Returns the initialized model.
//     pub fn init<B: Backend>(&self) -> Model<B> {
//         Model {
//             //커널 크기 3사용
//             //채널 1에서 8로 확장
//             // conv1: Conv2dConfig::new([1, 8], [3, 3]).init(),
//             // //8에서 16으로 확장
//             // conv2: Conv2dConfig::new([8, 16], [3, 3]).init(),
//             //적응형 평균 폴링 모듈을 사용 이미지의 차원을 8x8으로 축소
//             pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
//             activation: ReLU::new(),
//             linear1: LinearConfig::new(1, self.hidden_size).init(),
//             linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
//             dropout: DropoutConfig::new(self.dropout).init(),
//         }
//     }
//     pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
//         Model {
//             // conv1: Conv2dConfig::new([1, 8], [3, 3]).init_with(record.conv1),
//             // conv2: Conv2dConfig::new([8, 16], [3, 3]).init_with(record.conv2),
//             pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
//             activation: ReLU::new(),
//             linear1: LinearConfig::new(1, self.hidden_size).init_with(record.linear1),
//             linear2: LinearConfig::new(self.hidden_size, self.num_classes)
//                 .init_with(record.linear2),
//             dropout: DropoutConfig::new(self.dropout).init(),
//         }
//     }
// }

// /*
// data
// Batcher 구조체를 정의 => 텐서가 모델에 전달되기 전에  전송되어야 하는장치

// */



// pub struct Tester<B: Backend> {
//     device: B::Device,
// }

// impl<B: Backend> Tester<B> {
//     pub fn new(device: B::Device) -> Self {
//         Self { device }
//     }
// }

// #[derive(Clone, Debug)]
// pub struct Test<B: Backend> {
//     pub widths: Tensor<B, 1>,
//     pub targets: Tensor<B, 1, Int>,
// }

// impl<B: Backend> Batcher<DiabetesPatient, Test<B>> for Tester<B> {
//     fn batch(&self, items: Vec<DiabetesPatient>) -> Test<B> {
//         let widths = items
//         .iter()
//         .map(|item| Tensor::<B, 1>::from_data(Data::from([(item.width as f64).elem()])))
//         .collect();
//         let targets = items
//             .iter()
//             .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.height as i64).elem()])))
//             .collect();

//         let widths = Tensor::cat(widths, 0).to_device(&self.device);
//         let targets = Tensor::cat(targets, 0).to_device(&self.device);

//         Test { widths, targets }
//     }
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
//         .build(DiabetesDataset::new().unwrap());

//     let dataloader_test = DataLoaderBuilder::new(batcher_valid)
//         .batch_size(config.batch_size)
//         .shuffle(config.seed)
//         .num_workers(config.num_workers)
//         .build(DiabetesDataset::new().unwrap());

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
// pub fn main(){
//     DiabetesDataset::new();
//     let config = ModelConfig::new(10, 1);
//      println!("{}",config);
//      type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
//      type MyAutodiffBackend = Autodiff<MyBackend>;
//      let device = burn::backend::wgpu::WgpuDevice::default();
//     //학습
//     //  train::<MyAutodiffBackend>(
//     //     "./train",
//     //     TrainingConfig::new(ModelConfig::new(10, 1), AdamConfig::new()),
//     //     device,
//     // );


//     // infer::<MyAutodiffBackend >("./train",device,DiabetesPatient{width:1,height:1})
// }

// // pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: DiabetesPatient) {
// //     let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
// //         .expect("Config should exist for the model");
// //     let record = CompactRecorder::new()
// //         .load(format!("{artifact_dir}/model").into())
// //         .expect("Trained model should exist");

// //     let model = config.model.init_with::<B>(record).to_device(&device);

// //     let label = item.height;
// //     let batcher = Tester::new(device);
// //     let batch = batcher.batch(vec![item]);
// //     let output = model.forward(batch.widths);

// //     println!("{}",output.to_data());
// //     let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();
// //     //예측값과 실제 레이블값
// //     //학습이 이상하게 댐
// //     println!("Predicted {} Expected {}", predicted, label);
// // }
