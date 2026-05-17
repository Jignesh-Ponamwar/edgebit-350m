# Challenges of Quantization-Aware Training

Quantization-aware training (QAT) sounds straightforward: quantize weights during training so the model learns to work with reduced precision. In practice, it is one of the harder problems in deep learning engineering.

This post covers the specific challenges we encountered training EdgeBit-350M and how we addressed them.

## Challenge 1: The Gradient Gap

The fundamental problem of QAT is that quantization is non-differentiable. The `round()` function has zero gradient everywhere.

The Straight-Through Estimator (STE) approximates the gradient by pretending quantization did not happen:

```
Forward:   w_q = round(w / scale)    # discrete operation
Backward:  ∂L/∂w ≈ ∂L/∂w_q          # ignore the rounding
```

This is not a good approximation. The true gradient is zero, and we are using a non-zero approximation. Why does it work?

The key insight is that we are not trying to compute the exact gradient. We are trying to find a direction that improves the loss. The STE provides a useful direction: if the loss would decrease by making a weight larger, the STE gradient points in that direction, even though the actual weight value is quantized.

Over many steps, the full-precision shadow weights drift toward values that are naturally close to quantization boundaries. The STE gradient is noisier than a true gradient, but it is biased in a useful direction.

**What can go wrong**: If the learning rate is too high, the STE noise can dominate the useful signal. The model oscillates instead of converging. This is why we use conservative learning rates (3e-4) for ternary training.

## Challenge 2: The Scale Factor Problem

In ternary quantization, each weight group has a scale factor:

```
W_ternary = round(W / scale)
reconstructed_W = W_ternary * scale
```

The scale factor must be part of the forward pass (so the gradient can flow through it), but it must also be stable (so it does not oscillate or explode).

We use the absmean scale: `scale = mean(|W|)` per group. This has a useful property: as the model trains toward ternary values, the scale converges to a stable value that represents the typical weight magnitude.

**What can go wrong**: If the scale is detached from the computation graph (a common bug), gradients do not flow through it. The model can still train, but the scale never adapts, leading to a systematic bias. We spent an afternoon debugging this exact issue.

**Another failure mode**: If the scale approaches zero (because all weights in a group are near zero), division by scale produces very large values, which produce NaN after rounding. We clamp scales to a minimum of 1e-8.

## Challenge 3: Phase Transitions

The progressive curriculum transitions the model through four precision levels. Each transition is a controlled disruption:

```
BF16 → INT8:     Loss spike ~10-20%   (mild)
INT8 → INT4:     Loss spike ~30-50%   (moderate)
INT4 → Ternary:  Loss spike ~50-100%  (severe)
```

The INT4 → Ternary transition is particularly dangerous because:
1. The model goes from 16 quantization levels to 3
2. Many weights that were represented by distinct INT4 values collapse to the same ternary value
3. The effective model capacity drops sharply
4. Gradient norms spike as the loss landscape changes suddenly

**What works**:
- Gradient clipping at 1.0 (absorbs the spike)
- Not reducing learning rate (the model needs to adapt quickly)
- Giving the ternary phase 45% of total training (recovery takes time)
- SubLN (prevents activation magnitude drift during the transition)

**What does not work**:
- Gradual mixing of quantization modes (the discontinuity is inherent)
- Annealing the learning rate before transition (slows adaptation)
- Very short phases (the model does not have time to adapt)

## Challenge 4: Attention Instability

In full-precision training, attention logits are naturally bounded by the weight magnitudes and the softmax temperature. In ternary networks, this natural bound is weaker.

The problem manifests as:
1. Q and K values grow over training
2. Attention logits (Q @ K^T) grow quadratically
3. Softmax saturates (one attention weight approaches 1.0, rest approach 0.0)
4. Gradients through softmax vanish (saturated softmax has near-zero gradient)
5. The model stops learning

Without intervention, this happens reliably around step 2000-5000 of ternary training in a 24-layer model.

**Solution**: QK RMSNorm normalizes Q and K independently, keeping their magnitudes bounded regardless of weight values. The learnable scale factors allow the model to control attention sharpness without unbounded growth.

## Challenge 5: The Zero-Weight Trap

In ternary quantization, weights that are close to zero get quantized to exactly zero. This means they contribute nothing to the output and receive zero gradient through the STE.

If too many weights become zero, the model loses capacity. We have observed models where > 80% of weights are zero — effectively a 60M parameter model disguised as a 350M one.

**What causes this**: Aggressive weight decay combined with ternary quantization. Weight decay pushes weights toward zero, and ternary quantization snaps them to exactly zero.

**Mitigation**: We use moderate weight decay (0.01) and monitor the zero-weight percentage. A healthy ternary model has 50-60% zero weights. Above 70% is a warning sign.

## Challenge 6: Layer-Dependent Sensitivity

Not all layers are equally sensitive to quantization. In general:
- The first 2-3 layers are most sensitive (initial feature extraction)
- The last 2-3 layers are moderately sensitive (output refinement)
- The middle layers are least sensitive (redundant representations)

This means a uniform quantization policy is suboptimal. The first and last layers would benefit from higher precision.

**Current approach**: We use uniform ternary quantization for simplicity. The progressive curriculum partially addresses this by letting all layers adapt simultaneously.

**Future direction**: Mixed-precision per layer. Keep the first/last layers at INT8, middle layers at ternary. This requires architecture search or sensitivity analysis.

## Challenge 7: Reproducibility

QAT training is less reproducible than standard training. The quantization function introduces discontinuities that amplify small numerical differences:

- Different GPU architectures may produce different rounding results
- Different CUDA versions may produce different matmul results
- The STE gradient is an approximation, so small perturbations accumulate

**Practical impact**: Two training runs with identical hyperparameters on different hardware may produce models with different quality. We address this by:
- Setting deterministic seeds everywhere
- Reporting results averaged over multiple runs (when compute allows)
- Focusing on relative comparisons (ternary vs. FP16 on the same data) rather than absolute numbers

## The Meta-Challenge: Knowing What You Do Not Know

The hardest part of QAT engineering is distinguishing between:
1. A bug in the implementation
2. A fundamental limitation of the approach
3. A hyperparameter that needs tuning
4. Normal training variance

When the loss is higher than expected, is it because ternary weights are inherently less capable? Or because the learning rate is wrong? Or because there is a bug in the scale factor computation?

The only reliable answer is ablation. Train the model with and without each component, at multiple scales, on the same data. This is expensive but necessary.

Our approach: validate everything at the tiny model scale first. If a technique does not work at 50M parameters, it probably will not work at 350M either. But if it works at 50M, it might still fail at 350M for scale-dependent reasons. There is no shortcut.

## Summary of Practical Advice

1. Always start with a tiny model smoke test
2. Monitor gradient norms, weight distributions, and scale factors
3. Use gradient clipping (1.0 is a safe default)
4. Do not skip the progressive curriculum
5. SubLN and QK norm are requirements, not enhancements
6. Keep weight decay moderate (0.01)
7. Give the ternary phase at least 40% of total training time
8. Monitor zero-weight percentage
9. Test on multiple hardware configurations if reproducibility matters
10. When in doubt, train a baseline FP16 model and compare
