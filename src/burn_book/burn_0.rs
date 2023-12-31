use burn::tensor::Tensor;
use burn::backend::Wgpu;

type Backend = Wgpu;//낮은 수준의 작업

/*
vector = [1,1]
matrix= [[1][2]]
3d-tensor [[[]]] rank=3
?d-tendor

*/

pub fn main() {
    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]]);
    let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);//1로 채워진

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    println!("{}", tensor_1 + tensor_2);
}