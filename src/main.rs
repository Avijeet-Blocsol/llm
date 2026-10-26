#![recursion_limit = "256"]

mod experiments;
mod model;
mod tokenizer;

use burn::backend::{Autodiff, Wgpu};
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::cast::ToElement;
use burn::tensor::{Int, Tensor};

use model::GPTModel;
use tokenizer::Tokenizer;

fn main() {
    println!("=== Custom Rust GPT: End-to-End Pipeline ===");

    type MyBackend = Autodiff<Wgpu>;
    let device = Default::default();

    // 1. Train BPE Tokenizer
    let corpus = "the cat sat on the mat. the dog sat on the rug. the cat sat on the mat. the cat sat on the mat. the dog sat on the rug. the cat sat on the mat. the cat sat on the mat. the dog sat on the rug. the cat sat on the mat. the cat sat on the mat. the dog sat on the rug. the cat sat on the mat. the cat sat on the mat. the dog sat on the rug. the cat sat on the mat.";
    let mut tokenizer = Tokenizer::new();
    tokenizer.train(corpus, 270);

    let vocab_size = tokenizer.vocab.len();
    println!("Trained vocabulary size: {}", vocab_size);

    let tokens = tokenizer.encode(corpus);
    println!("Encoded corpus into {} tokens", tokens.len());

    // 2. Prepare Training Sequence (Input X vs Target Y)
    let seq_len = 8;
    assert!(tokens.len() > seq_len, "Corpus is too short for seq_len");

    let x_data: Vec<i64> = tokens[0..seq_len].iter().map(|&t| t as i64).collect();
    let y_data: Vec<i64> = tokens[1..=seq_len].iter().map(|&t| t as i64).collect();

    let x = Tensor::<MyBackend, 1, Int>::from_ints(x_data.as_slice(), &device).unsqueeze::<2>();
    let y = Tensor::<MyBackend, 1, Int>::from_ints(y_data.as_slice(), &device);

    // 3. Initialize Model and Training Components
    let max_seq_len = 64;
    let d_model = 32;
    let n_heads = 4;
    let n_layers = 2;

    println!("Initializing GPT-3 Architecture on GPU...");
    let mut model = GPTModel::<MyBackend>::new(
        vocab_size,
        max_seq_len,
        d_model,
        n_heads,
        n_layers,
        &device,
    );

    let loss_fn: CrossEntropyLoss<MyBackend> = CrossEntropyLossConfig::new().init(&device);
    let mut optimizer = AdamWConfig::new().init::<MyBackend, GPTModel<MyBackend>>();
    let learning_rate = 5e-3;

    // 4. Training Loop (Watch loss decrease)
    println!("\n--- Training Model ---");
    let total_epochs = 60;
    for epoch in 1..=total_epochs {
        let logits = model.forward(x.clone());
        let loss = loss_fn.forward(logits.reshape([seq_len, vocab_size]), y.clone());

        let loss_val: f32 = loss.clone().into_data().as_slice::<f32>().unwrap()[0];

        if epoch == 1 || epoch % 10 == 0 {
            println!("Epoch {:2}/{} | Loss: {:.4}", epoch, total_epochs, loss_val);
        }

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(learning_rate, model, grads);
    }

    // 5. Autoregressive Text Generation
    println!("\n--- Generating Text ---");
    let prompt = "the cat";
    let mut gen_tokens = tokenizer.encode(prompt);
    let gen_length = 10;

    for _ in 0..gen_length {
        let curr_len = gen_tokens.len();
        let context_slice = if curr_len > max_seq_len {
            &gen_tokens[curr_len - max_seq_len..]
        } else {
            &gen_tokens[..]
        };

        let input_vec: Vec<i64> = context_slice.iter().map(|&t| t as i64).collect();
        let input_tensor =
            Tensor::<MyBackend, 1, Int>::from_ints(input_vec.as_slice(), &device).unsqueeze::<2>();

        let logits = model.forward(input_tensor);
        let [_b, s, v] = logits.dims();

        // Extract logits of the very last position
        let last_logit = logits.slice([0..1, (s - 1)..s, 0..v]).reshape([v]);
        let next_token_id = last_logit.argmax(0).into_scalar().to_usize();

        gen_tokens.push(next_token_id);
    }

    let generated_text = tokenizer.decode(&gen_tokens);
    println!("Prompt:    \"{}\"", prompt);
    println!("Generated: \"{}\"", generated_text);
    println!("\nAll phases completed successfully!");
}
