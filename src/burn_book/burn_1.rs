use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d,Conv2dConfig},
    }
};

pub struct ModelConfig{

}
impl ModelConfig {

}

impl<B:Backend> Model<B>{
    pub fn forward(&self,imges:Tensor<B,3>)->Tensor<B,2>{
        
    }
}
fn main(){

}