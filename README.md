# shifted_nanoGPT
nanoGPT but with the activation functions randomly shifted.

## Code

```py
class ShiftedReLU(nn.Module):
    """ReLU with fixed per-neuron random shifts near zero (like weight init)."""
    def __init__(self, num_neurons):
        super().__init__()
        # Shifts sampled once from N(0, 0.02), same as Linear weight init in this model
        shifts = torch.randn(num_neurons) * 0.02
        self.register_buffer('shift', shifts)  # non-trainable

    def forward(self, x):
        return F.relu(x - self.shift)
```

## Usage

```
git clone karpathy's nanoGPT and replace model.py with the one in this repo, then run with the following command:
python3.10 train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

## Results

baseline:
```
step 2000: train loss 1.7648, val loss 1.8857
```
shifted across 3 seeds:
```
step 2000: train loss 1.6917, val loss 1.8424
step 2000: train loss 1.6530, val loss 1.8168
step 2000: train loss 1.6735, val loss 1.8473
```
