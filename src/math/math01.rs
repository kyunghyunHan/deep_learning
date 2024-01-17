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
    let  d: Array<i64, _> = Array::ones((2, 2)).into_dyn();
    println!("{}", d);
    /*reshape */
    let e = d.clone().into_shape((1, 4)).unwrap();
    println!("{}", e);
    let f = d.clone().into_shape((1, 4)).unwrap();
   
    println!("{}", f);

    let g= arr2(&[[0,1,2,3],[4,5,6,7],[8,9,10,11]]);
    let first_row_slice = g.slice(s![1, ..]);
    println!("{:?}", first_row_slice);
    let sliced_a = g.slice(s![..2,..3]);
    println!("{}", sliced_a);

    let a = Array::range(0., 24., 1.).into_shape((4, 3,2)).unwrap();;

    println!("{:?}", a);
}