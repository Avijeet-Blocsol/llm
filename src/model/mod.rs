use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::{Backend, Device};
use burn::tensor::{Float, Int, Tensor};

#[derive(Module, Debug)]
pub struct GPTEmbeddings <B: Backend> {
    pub token_embedding: Embedding<B>,
    pub position_embedding: Embedding<B>,
}

impl <B: Backend> GPTEmbeddings <B> {
    pub fn new (vocab_length: usize, max_sequence_length: usize, model_dimensions: usize, device: &B::Device) -> GPTEmbeddings<B> {
        let token_emb = EmbeddingConfig::new(vocab_length, model_dimensions).init(device);
        let pos_emb = EmbeddingConfig::new(max_sequence_length, model_dimensions).init(device);

        return Self {
            token_embedding: token_emb,
            position_embedding: pos_emb,
        }
    }

    pub fn forward (&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batch_len, max_seq_size] = tokens.dims();

        let device = tokens.device();

        let token_embs = self.token_embedding.forward(tokens);

        let positions = Tensor::<B, 1, Int>::arange(0..max_seq_size as i64, &device).unsqueeze();

        let position_embs = self.position_embedding.forward(positions);

        return token_embs + position_embs;
    }
}

#[derive(Module, Debug)]
pub struct CasualSelfAttention <B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub out_proj: Linear<B>,
    pub n_heads: usize,
    pub d_model: usize,
}

impl <B: Backend> CasualSelfAttention<B> {
    pub fn new(d_model: usize, n_heads: usize, device: &B::Device ) -> Self {
        
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");

        return Self {
            q_proj: LinearConfig::new(d_model, d_model).init(device),
            k_proj: LinearConfig::new(d_model, d_model).init(device),
            v_proj: LinearConfig::new(d_model, d_model).init(device),
            out_proj: LinearConfig::new(d_model, d_model).init(device),
            n_heads: n_heads,
            d_model: d_model,
        };
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        // First we perform matrix multiplication ith the weight matrices to form the Q, K and V matrices

        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Next, for multihead attention, we need to reshape these matrices such that d_dims becomes n_heads and head_dims
        // We swap the dims because when we run matrix multiplication, only the last 2 items matter to give the result
        // We need to calculate the score over all the attention heads at once.
        let head_dims = self.d_model / self.n_heads;

        let q = q.reshape([batch_size, seq_len, self.n_heads, head_dims]).swap_dims(1, 2);
        let k = k.reshape([batch_size, seq_len, self.n_heads, head_dims]).swap_dims(1, 2);
        let v = v.reshape([batch_size, seq_len, self.n_heads, head_dims]).swap_dims(1, 2);

        // Now we have the right matrices, lets calculate the scores. We swap the dims of K because we need to perform a 
        // matrix transporse. We divide the values by square root of head dims for normalization.
        let scores = q.matmul(k.swap_dims(2, 3)) / (head_dims as f32).sqrt();

        // Since now we have the scores, we need to apply a mask. This is simply because we will reject any values for a token
        // which exist in the forward sequence of a token. In case we had a bidirectional encoding, we would have not applied any
        // mask here.
        let device = scores.device();
        let row_idx = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device).reshape([seq_len, 1]);
        let col_idx = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device).reshape([1, seq_len]);
        let mask = col_idx.greater(row_idx).reshape([1, 1, seq_len, seq_len]);
        

        let masked_scores = scores.mask_fill(mask, -1e9);
        let weights = softmax(masked_scores, 3);

        let context = weights.matmul(v);

        let merged = context.swap_dims(1,2).reshape([batch_size, seq_len, self.d_model]);

        return self.out_proj.forward(merged);
    }
}

#[derive(Module, Debug)]
pub struct GPTMlp <B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl <B: Backend> GPTMlp<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Self {
        return Self {
            fc1: LinearConfig::new(d_model, 4*d_model).init(device),
            fc2: LinearConfig::new(4*d_model, d_model).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = gelu(self.fc1.forward(x));
        return self.fc2.forward(hidden);
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock <B: Backend> {
    attention: CasualSelfAttention<B>,
    mlp: GPTMlp<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>
}

impl <B: Backend> TransformerBlock<B> {
    pub fn new(d_model: usize, n_heads: usize, device: &B::Device) -> Self {
        let att = CasualSelfAttention::new(d_model, n_heads, device);
        let mlp = GPTMlp::new(d_model, device);

        let norm1 = LayerNormConfig::new(d_model).init(device);
        let norm2 = LayerNormConfig::new(d_model).init(device);

        return Self {
            attention: att,
            mlp: mlp,
            norm1: norm1,
            norm2: norm2,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attention.forward(self.norm1.forward(x.clone()));
        return x.clone() + self.mlp.forward(self.norm2.forward(x));
    }
}

#[derive(Module, Debug)]
pub struct GPTModel <B: Backend> {
    embeddings: GPTEmbeddings<B>,
    blocs: Vec<TransformerBlock<B>>,
    l_norm: LayerNorm<B>,
    f_out: Linear<B>
}

impl <B: Backend> GPTModel<B> {
    pub fn new(vocab_length: usize, max_sequence_length: usize, d_model: usize, n_heads: usize, n_layers: usize, device: &B::Device) -> Self {
        let embeds = GPTEmbeddings::new(vocab_length, max_sequence_length, d_model, device);

        let mut blocks: Vec<TransformerBlock<B>> = Vec::new();

        for _i in 0..n_layers {
            blocks.push(TransformerBlock::new(d_model, n_heads, device));
        }

        let l_norm = LayerNormConfig::new(d_model).init(device);
        let f_out = LinearConfig::new(d_model, vocab_length).init(device);

        return Self {
            embeddings: embeds,
            blocs: blocks,
            l_norm: l_norm,
            f_out: f_out,
        }
    }

    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.embeddings.forward(x.clone());

        for i in 0..self.blocs.len() {
            x = self.blocs[i].forward(x);
        }

        let l_norm = self.l_norm.forward(x);

        let out = self.f_out.forward(l_norm);

        return out;
    }
}