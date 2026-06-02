# Language Model Pretrain

This project provides an example implementation for training next-token prediction
models on MiniPile datasets using the Burn-based RWKV deep learning library.

## Dataset Details

- [The MiniPile Challenge for Data-Efficient Language Models](https://arxiv.org/abs/2304.08442)
- MiniPile is a 6GB subset of the [deduplicated The Pile corpus](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated).
- More details on the MiniPile curation procedure and pre-training results are in the [MiniPile paper](https://arxiv.org/abs/2304.08442).
- For more details on the Pile corpus, see [the Pile datasheet](https://arxiv.org/abs/2201.07311).
- You can download the dataset from [BlinkDL/minipile-tokenized](https://huggingface.co/datasets/BlinkDL/minipile-tokenized).

## CUDA Backend

```bash
cargo run -p rwkv-lm-train --example train --release --features cuda
```

## WGPU Backend

```bash
cargo run -p rwkv-lm-train --example train --release --features wgpu
```

## Metal Backend

```bash
cargo run -p rwkv-lm-train --example train --release --features metal
```
