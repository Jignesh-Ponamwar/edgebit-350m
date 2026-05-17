# Building a Ternary Transformer

## The Core Idea

What if every weight in a neural network could only be -1, 0, or +1?

This sounds absurd. Neural networks work because they learn precise, continuous weight values through gradient descent. Reducing weights to three values throws away almost all of that learned precision.

But it also reduces memory by 16x (2 bits per weight instead of 32) and replaces floating-point multiplication with integer addition. For edge deployment, this tradeoff is worth investigating.

## The BitLinear Layer

The foundation of EdgeBit is the BitLinear layer, which replaces `nn.Linear` throughout the transformer.

During the forward pass, BitLinear does three things:

**1. Quantize the weights to ternary:**
```
scale = mean(|W|)  per group of 128
W_ternary = clamp(round(W / scale), -1, +1)
```

The absmean scale captures the magnitude. The rounding maps each weight to its nearest ternary value. Clamping ensures we stay in {-1, 0, +1}.

**2. Scale the activations to INT8:**
```
scale_x = max(|x|) / 127
x_int8 = round(x / scale_x) * scale_x
```

This bounds the activation range for numerical stability.

**3. Compute the output and rescale:**
```
y = x_int8 @ W_ternary.T
y = y * (scale_x * scale_w)
```

The output is a real-valued tensor. The quantization happens inside the forward pass and is invisible to the rest of the model.

## The Gradient Problem

Here is the fundamental challenge: the `round()` function has zero gradient everywhere (except at integers, where it is undefined). Standard backpropagation cannot flow through it.

The solution is the Straight-Through Estimator (STE). During the backward pass, we pretend the quantization function is the identity:

```
Forward:   w_q = round(w / scale)     # discrete
Backward:  ∂L/∂w = ∂L/∂w_q            # pretend round() wasn't there
```

The model maintains full-precision "shadow" weights. Each forward pass quantizes them to ternary. Each backward pass updates the shadow weights in full precision. Over time, the shadow weights drift toward values that are naturally close to ternary.

This is not a perfect gradient — it is an approximation. But it works well enough for training, especially with the stability techniques we add.

## Why Ternary Training Alone Is Not Enough

If you initialize a transformer with random weights and immediately train with ternary quantization, you get garbage. The model cannot learn from the chaotic gradient signal of random ternary weights.

This is why we use a progressive quantization curriculum:

1. **BF16 warmup** (10% of training): Train normally. The model learns basic features.
2. **INT8** (20%): Introduce mild quantization. The model adapts its weight distribution.
3. **INT4** (25%): Significant quantization. Weight distributions sharpen.
4. **Ternary** (45%): Target precision. Extended training for full convergence.

Each transition causes a loss spike. The model temporarily forgets some of what it learned. But because the transitions are gradual, the recovery is fast. By the time we reach ternary, the model has already learned to represent information in ways that survive quantization.

## Stabilization: The Details That Matter

Three additional techniques make ternary training work at depth:

**QK Normalization.** In standard transformers, attention logits are bounded by the hidden dimension and the softmax temperature. In ternary networks, Q and K values can grow unboundedly because the quantization removes the implicit normalization of full-precision weights. We apply RMSNorm independently to Q and K, with learnable scale factors, to keep attention logits in a safe range.

**SubLN.** Standard pre-norm transformers normalize the input to each sub-layer. SubLN adds normalization *inside* the sub-layer — specifically before the output projection. This prevents activation magnitudes from drifting in deep ternary networks. It is initialized as the identity function so it has no effect at the start of training.

**Grouped quantization.** A single scale factor for an entire weight matrix loses too much information. We divide weights into groups of 128 and compute independent scales per group. This adds minimal storage (1 float16 per 128 weights) but significantly improves representation.

## The Weight Distribution Story

As training progresses through the curriculum, the weight distribution tells a story:

**BF16 phase**: Weights are normally distributed, centered at zero.

**INT8 phase**: Distribution stays roughly normal but tails get clipped.

**INT4 phase**: Distribution sharpens. Peaks emerge near the 16 quantization levels.

**Ternary phase**: Distribution collapses to three peaks at -1, 0, +1. The zero peak is usually the tallest (many weights are pruned to zero). The scale factors capture what the magnitude would have been.

This is not pruning followed by quantization. It is a single training process that simultaneously learns which weights matter (non-zero) and what their sign should be.

## What We Give Up

Ternary quantization is not free. At matched parameter count, a ternary model is less capable than a full-precision model. The effective information capacity is lower because each weight carries less information.

A rough rule of thumb: a ternary model with N parameters has roughly the capacity of a full-precision model with N/4 to N/6 parameters. A 350M ternary model is comparable to a 60-90M FP16 model in terms of what it can represent.

But the 350M ternary model is also *much smaller* in memory. A 90M FP16 model needs ~180MB. The 350M ternary model needs ~35MB for weights (plus embeddings). The ternary model has more parameters doing less work per parameter, but in a smaller memory footprint.

Whether this tradeoff is worthwhile depends entirely on the deployment constraint. If you have unlimited memory, use full-precision. If you are deploying on a Raspberry Pi, ternary wins.

## Lessons Learned

1. **The curriculum matters more than the architecture.** The same architecture trained with different curriculum schedules produces wildly different results. Getting the phase ratios and transition timing right is critical.

2. **Stability techniques are not optional.** Without QK norm and SubLN, ternary training at 24 layers diverges reliably. These are not minor improvements — they are requirements.

3. **Grouped quantization is high-value, low-cost.** The memory overhead of group scales is negligible (< 1% of total), but the quality improvement is substantial.

4. **The STE works better than expected.** Despite being a crude approximation, the straight-through estimator produces useful gradients for ternary training. The key is giving it enough training time (the 45% ternary phase) to converge.

5. **The embedding table is the elephant in the room.** At 350M params, the embedding table is 46% of total parameters. NF4 quantization helps, but better embedding compression is the single highest-impact research direction.
