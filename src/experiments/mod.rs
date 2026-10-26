

use burn::prelude::*;

pub fn explore_tensors<B: Backend>() {
    let d = B::Device::default();

    let a: Tensor<B, 2, Int> = Tensor::from_ints([[1, 2], [3, 4]], &d);
    let b: Tensor<B, 2, Int> = Tensor::from_ints([[5, 6], [7, 8]], &d);

    let c = a.clone() + b.clone();
    let d = a.clone().matmul(b.clone());

    println!("a: {}", a.clone());
    println!("b: {}", b.clone());
    println!("c: {}", c.clone());
    println!("d: {}", d.clone());
}

pub fn explore_tensor_operations<B: Backend>() {
    let d = B::Device::default();

    let a: Tensor<B, 3, Float> = Tensor::from_floats([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], &d);

    println!("a is: {} ", a.clone());
    println!("Shape is: {:?}", a.clone().dims());

    let reshaped_a = a.clone().reshape([3, 2]);

    println!("Reshaped a is: {} ", reshaped_a.clone());
    println!("Reshaped shape is: {:?}", reshaped_a.clone().dims());

    let swapped_a = reshaped_a.clone().swap_dims(0, 1);

    println!("Swapped a is: {} ", swapped_a.clone());
    println!("Swapped shape is: {:?}", swapped_a.clone().dims());

    let swapped_b = reshaped_a.clone().reshape([3, 2, 1, 1, 1]);

    println!("Swapped b is: {} ", swapped_b.clone());
    println!("Swapped b shape is: {:?}", swapped_b.clone().dims());

    let res : Tensor<B, 2>= reshaped_a.matmul(swapped_a.clone());

    println!("Result of matmul is: {} ", res.clone());
    println!("Result shape is: {:?}", res.clone().dims());
}