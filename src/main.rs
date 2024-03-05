mod algorithm;
mod fast_deep;
mod activation_function;
mod university;
mod math;
mod burn_book;
use rayon::prelude::*;
use ndarray::prelude::*;
mod boston_price;
fn main(){
    
//    let mut arr= arr1(&[1,0]);

//    arr[0] =2;
//    println!("{}",arr);
    // algorithm::bottom_0::main();
    algorithm::bottom_2::main();
    // algorithm::two_layer_net::main();
    // boston_price::model::main();
    /*===============fast_depp================ */
    // fast_deep::fast_deep01::main();
    /*===============fast_depp================ */

    /*===============activation_function================ */
    // activation_function::sigmoid::main();
    // activation_function::tanh::main();
    /*===============activation_function================ */
    /*===============math================ */
    //  math::numpy::main();
    /*===============math================ */
    /*===============burn_book================ */
    //  burn_book::burn_1::main();
    // burn_book::custom::main();
    // math::math01::main();
    /*===============burn_book================ */



}

