use std::any::Any;

use ndarray::prelude::*;

pub fn main(){
    let a= arr1(&[1,2,3,4]);
 

    println!("{}",a.len());
    println!("{:?}",a.shape());
    println!("{:?}",a.t);


    let b= arr2(&[[1,2,3,4],[5,6,7,8]]);
    println!("{}",b);
    println!("{:?}",b.shape())
}