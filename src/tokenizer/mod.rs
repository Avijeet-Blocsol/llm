#[allow(dead_code)]

use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct Tokenizer {
    pub vocab: HashMap<Vec<u8>, usize>,
    pub inverse_vocab: HashMap<usize, Vec<u8>>,
    pub merges: HashMap<(usize, usize), (usize, usize)>,
}

impl Tokenizer {
    pub fn new () -> Tokenizer {

        let mut vcb: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut r_vcb: HashMap<usize, Vec<u8>> = HashMap::new();

        for i in 0..=255 {
            vcb.insert(vec![i.clone()], i.clone() as usize);
            r_vcb.insert(i.clone() as usize, vec![i]);
        }

        let eot = b"<|endoftext|>".to_vec();

        vcb.insert(eot.clone(), 256);
        r_vcb.insert(256, eot);

        return Tokenizer{
            vocab: vcb,
            inverse_vocab: r_vcb,
            merges: HashMap::new(),
        }
    }

    fn count_pairs (tokens: &[usize]) -> HashMap<(usize, usize), usize> {
        let mut result: HashMap<(usize, usize), usize> = HashMap::new();

        if tokens.len() < 2 {
            return result;
        }

        for i in 0..(tokens.len()-1) {
            *result.entry((tokens[i], tokens[i+1])).or_insert(0 as usize) += 1;
        }

        return result;
    }

    fn merge_tokens (tokens: &[usize], pair: (usize, usize), id: usize) -> Vec<usize> {
        let mut result: Vec<usize> = Vec::new();

        if tokens.len() < 2 {
            return tokens.to_vec();
        }

        let mut i = 0;

        while i < tokens.len() {
            if i < tokens.len() - 1 && tokens[i] == pair.0 && tokens[i+1] == pair.1 {
                result.push(id);
                i += 2;
            } else {
                result.push(tokens[i]);
                i += 1;
            }
        }

        return result;
    }

    pub fn train (&mut self, text: &str, target_vocab_size: usize) {
        let mut tokens: Vec<usize> = text.as_bytes().iter().map(|&b| b as usize).collect();
        let mut priority = 0;

        while self.vocab.len() < target_vocab_size {
            let pairs = Self::count_pairs(&tokens);

            let highest = pairs.iter().max_by_key(|&x| x.1);

            if highest.is_none() || Some(highest.unwrap().1) < Some(&2) {
                break;
            }

            let highest_pair = highest.unwrap().0;
            let new_id = self.vocab.len();

            let mut new_bytes = self.inverse_vocab[&highest_pair.0].clone();
            new_bytes.extend(&self.inverse_vocab[&highest_pair.1]);

            self.vocab.insert(new_bytes.clone(), new_id);
            self.inverse_vocab.insert(new_id, new_bytes);
            self.merges.insert(*highest_pair, (priority, new_id));
            
            tokens = Self::merge_tokens(&tokens, *highest_pair, new_id);
            priority += 1;
        }
    }

    pub fn decode (&self, tokens: &[usize]) -> String {
        let mut pieces: Vec<u8> = Vec::new();

        for t in tokens {
            let byte_vec = self.inverse_vocab[t].clone();

            for b in byte_vec {
                pieces.push(b);
            }
        }

        return String::from_utf8_lossy(&pieces).to_string();
    }
    
    pub fn encode (&self, text: &str) -> Vec<usize> {
        let mut result: Vec<usize> = text.as_bytes().iter().map(|&b| b as usize).collect();

        while result.len() > 1 {
            let mut best_pair: Option<((usize, usize), usize, usize)> = None;

            for i in 0..(result.len() - 1) {
                let pair = (result[i], result[i+1]);

                if let Some(&(priority, new_id)) = self.merges.get(&pair) {
                    match best_pair {
                        None => {
                            best_pair = Some((pair, priority, new_id));
                        },
                        Some(b) => {
                            if b.1 > priority {
                                best_pair = Some((pair, priority, new_id));
                            } 
                        }
                    }
                } 
            }

            match best_pair {
                None => {
                    break;
                }

                Some(b) => {
                    result = Self::merge_tokens(&result, b.0, b.2);
                }
            }
        }

        return result
    }

}

