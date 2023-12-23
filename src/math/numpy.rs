use std::any::Any;

use ndarray::prelude::*;

pub fn main(){
    let a= arr1(&[1,2,3,4]);
 

    println!("{}",a.len());
    println!("{:?}",a.shape());
    println!("{:?}",a.type_id());
}