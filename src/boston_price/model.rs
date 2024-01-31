use polars::prelude::*;
use std::fs::File;
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
        }, Data, Int, Tensor
    },
    train::{
        ClassificationOutput,TrainStep,TrainOutput,ValidStep,LearnerBuilder,
        metric::{
            AccuracyMetric,
            LossMetric
        }
    },
    optim::AdamConfig,
    data::dataloader::{DataLoaderBuilder,batcher::Batcher},
    record::CompactRecorder,
    backend::{
        Autodiff,
        Wgpu,
        wgpu::AutoGraphicsApi
    }
};
use rayon::prelude::*;
pub fn main(){
    let train_df= CsvReader::from_path("./datasets/digit-recognizer/train.csv").unwrap().finish().unwrap();
    println!("{}",train_df);
}