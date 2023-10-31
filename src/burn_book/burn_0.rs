use burn::tensor::Tensor;
use burn::backend::WgpuBackend;
use burn::tensor::backend::Backend;


// Type alias for the backend to use.


fn main() {
    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    let tensor_1 = Tensor::<WgpuBackend, 2>::from_data([[2., 3.], [4., 5.]]);
    let tensor_2 = Tensor::<WgpuBackend, 2>::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    println!("{}", tensor_1 + tensor_2);
}