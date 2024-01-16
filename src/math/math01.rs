use std::any::Any;

use ndarray::prelude::*;

pub fn main(){
    /*배열정의 */
    let a= arr1(&[1,2,3,4]);
 

    println!("{}",a.len());
    println!("{:?}",a.shape());
    /*2차원 배열 */
    let b= arr2(&[[1,2,3,4],[5,6,7,8]]);
    println!("{}",b);
    println!("{:?}",b.shape());

    /*0배열과1배열
    
     */
    //2행 2열
    let c:Array::<f64, _>=Array::zeros((2,2).f());
    println!("{}",c);
    /*2행 2열 */
    let mut d: Array<i64, _> = Array::ones((2, 2)).into_dyn();
    println!("{}", d);

    let cc = d.into_shape((1, 4)).unwrap();
    println!("{}", cc);

 


}