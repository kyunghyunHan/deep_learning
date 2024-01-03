/*
data


*/


use burn::{
    data::{dataloader::batcher::Batcher, dataset::source::huggingface::MNISTItem},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};
pub struct MNISTBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MNISTBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}
pub fn main(){

}