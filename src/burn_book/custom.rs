use burn::data::dataset::{Dataset, InMemDataset};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::copy,
    path::{Path, PathBuf},
};

use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},//기본convolution
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, ReLU,
        loss::CrossEntropyLoss
    },
    tensor::{
        backend::{
            Backend,AutodiffBackend
        }, Data, ElementConversion, Int, Tensor
    },
    train::{
        ClassificationOutput,TrainStep,TrainOutput,ValidStep,LearnerBuilder,
        metric::{
            AccuracyMetric,
            LossMetric
        }
    },
    optim::AdamConfig,
    data::{
        dataloader::{
        DataLoaderBuilder,batcher::Batcher
        },
        dataset::source::huggingface::{
            MNISTDataset,MNISTItem
        }
    },
    record::CompactRecorder,
    backend::{
        Autodiff,
        Wgpu,
        wgpu::AutoGraphicsApi
    }
};
// type Backend = Wgpu;//낮은 수준의 작업
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct DiabetesPatient {
    #[serde(rename = "Weight")]
    pub weoght: f32,

    #[serde(rename = "Height")]
    pub height: f32,


}

pub struct DiabetesDataset {
    dataset: InMemDataset<DiabetesPatient>,
}

impl DiabetesDataset {
    pub fn new() -> Result<Self, std::io::Error> {
        let example_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
        let path = example_dir.join("../dataset/test.csv");

        let dataset = InMemDataset::from_csv(path).unwrap();
        let dataset = Self { dataset };
        Ok(dataset)
    }
}

// Implement the `Dataset` trait which requires `get` and `len`
impl Dataset<DiabetesPatient> for DiabetesDataset {
    fn get(&self, index: usize) -> Option<DiabetesPatient> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}
#[derive(Module, Debug)]//딥러닝 모듈생성
pub struct Model<B: Backend> {//BackEnd:새모델이 모든 벡엔드에서 실행할수 있게함
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: ReLU,
}

pub struct TESTBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TESTBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct TestBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<TestItem, TestBatch<B>> for TESTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> TestBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            // Normalize: make between [0,1] and make the mean=0 and std=1
            // values mean=0.1307,std=0.3081 are from the PyTorch MNIST example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as i64).elem()])))
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        MNISTBatch { images, targets }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);


        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}
pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = TESTBatcher::<B>::new(device.clone());
    let batcher_valid = TESTBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(DiabetesDataset::new().unwrap());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(DiabetesDataset::new().unwrap());

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
pub fn main() {
    let dataset = DiabetesDataset::new().expect("Could not load diabetes dataset");

    println!("Dataset loaded with {} rows", dataset.len());

    // Display first and last elements
    let item = dataset.get(0).unwrap();
    println!("First item:\n{:?}", item);

    let item = dataset.get(1).unwrap();
    println!("Last item:\n{:?}", item);
}