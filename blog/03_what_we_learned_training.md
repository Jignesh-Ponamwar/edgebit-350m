# What We Learned Training EdgeBit-350M

This post is a collection of practical lessons from training a ternary transformer. Some are well-known in the quantization literature. Some surprised us.

## Lesson 1: Start Small, Then Scale

We did not train the 350M model first. We trained three models:

1. **Tiny** (50M params, 5K steps): Validated convergence and the curriculum pipeline
2. **Small** (125M params, 15K steps): Validated scaling behavior
3. **Base** (350M params, 50K steps): Full training

The tiny model caught several bugs that would have been expensive to find at scale:
- A misaligned dimension in the GQA repeat logic
- A quantization scale that was not being detached from the computation graph
- A curriculum transition that was off-by-one in the step counting

Each bug would have wasted hours of A100 time at the 350M scale. At the tiny scale, they were caught in minutes.

**Takeaway**: Never skip the small-scale validation step.

## Lesson 2: Phase Transitions Are the Critical Moments

The progressive curriculum has four phases: BF16 warmup, INT8, INT4, ternary. Each transition is a moment of risk.

What happens at a transition:
1. Every BitLinear layer simultaneously changes its quantization mode
2. The effective weight values change discontinuously
3. The loss spikes because the model's representations are suddenly perturbed
4. Gradients temporarily become large and noisy
5. The model recovers over the next few hundred steps

The INT4 → ternary transition is the most dangerous. The model goes from 16 quantization levels to 3. If the learning rate is too high, the gradient noise from the transition can push the model into a bad region it cannot recover from.

**What worked**:
- Gradient clipping at 1.0 (essential, not optional)
- Monitoring gradient norms around transitions
- Giving the ternary phase 45% of total training (it needs the time)
- Not reducing learning rate at transitions (the model adapts faster at the current LR)

**What did not work**:
- Abrupt transitions without curriculum (loss diverges)
- Transition with very high LR (> 1e-3) at the ternary phase
- Extremely short phases (< 5% of training each)

## Lesson 3: Weight Distributions Tell You Everything

The most informative diagnostic during training is the weight distribution histogram.

**Healthy BF16 phase**: Normal distribution, centered at zero, tails extending to +-0.3.

**Healthy INT8 phase**: Similar to BF16 but with slightly sharper peaks. The tails may be compressed.

**Healthy INT4 phase**: Distinct peaks at the 16 quantization levels. The distribution starts to look like a histogram of a histogram.

**Healthy ternary phase**: Three clear peaks at -1, 0, +1. The zero peak is typically the tallest (50-70% of weights). The +-1 peaks are roughly symmetric.

**Warning signs**:
- All weights collapsing to zero (the model has given up)
- One of the +-1 peaks being much larger than the other (asymmetric learning)
- Weights not converging to the ternary peaks after 1000+ ternary steps
- Scale factors growing unboundedly (indicates the ternary values are compensating for something)

## Lesson 4: The Embedding Table Is Special

The embedding table (151,936 x 1024 = 155M parameters) is 46% of the model. It cannot use ternary quantization because embeddings need fine-grained distinctions between tokens.

We use NF4 (4-bit NormalFloat) quantization for embeddings. This is a compromise:
- 4 bits per value (8x compression vs FP32, 4x vs FP16)
- Non-uniform quantization levels optimized for normal distributions
- Acceptable reconstruction error (< 5% relative)

The embedding table is the single largest memory consumer in the packed model (82MB out of 156MB total). Better embedding compression — potentially learned codebooks or hash embeddings — is the highest-leverage optimization for future versions.

## Lesson 5: KV Cache Quantization Is Essentially Free

INT8 quantization of the KV cache halves its memory with negligible quality impact. We measured reconstruction error at < 0.5% relative.

The reason: attention values are already normalized (post-softmax weights sum to 1). INT8 has sufficient dynamic range to represent these well-behaved values.

At 2048-token context with 24 layers and 4 KV heads:
- FP16 KV cache: 24 MB
- INT8 KV cache: 12 MB

This 12MB savings matters on a 4GB device.

## Lesson 6: SubLN Matters More Than We Expected

We initially included SubLN (normalization before the output projection) as a minor stability improvement. It turned out to be critical.

Without SubLN, ternary training at 24 layers produces loss NaN within the first 1000 ternary steps. With SubLN, training is stable.

The reason: ternary weights produce output magnitudes that vary wildly depending on the input distribution. Without normalization before the output projection, these magnitude variations compound across layers. By layer 24, the activations can be arbitrarily large or small.

SubLN is initialized as identity (gamma=1), so it does nothing at the start of training. As the model trains through the quantization curriculum, SubLN learns to normalize the internal representations, preventing magnitude drift.

## Lesson 7: Data Quality Matters Even at Small Scale

With limited training budget, data quality has outsized impact. We found that:
- Deduplication mattered: repeated sequences in the training data led to memorization rather than generalization
- Minimum text length mattered: very short documents (< 20 characters) degraded training stability
- Domain diversity mattered: training on a single domain (e.g., only code or only news) produced a model that could not generalize

For a 350M model trained on 5-8B tokens, we recommend:
- At least 3 different data domains
- Deduplication at the document level
- Minimum document length of 50 tokens
- Shuffling across domains (not sequential domain training)

## Lesson 8: Smoke Tests Save Hours

The `--smoke_test` flag generates 200 synthetic samples and trains for 50 steps. This catches:
- Import errors
- Shape mismatches
- Device placement bugs
- Data loading issues
- Checkpoint save/load bugs

Running the smoke test before every real training job saved us from multiple wasted A100 hours.

## Lesson 9: The Monitoring Script Is Not Optional

GPU utilization drops, training stalls, disk fills up, pods disconnect. Without monitoring, you discover these problems when you check back hours later and find your training has been idle.

The monitoring script checks:
- GPU temperature and utilization
- Checkpoint age (alert if no new checkpoint in 2 hours)
- Disk usage (alert at 90%)
- Optional webhook alerts to Slack/Discord

Running the monitor in a separate tmux pane is standard practice.

## Lesson 10: The Model Is the Least Important Part

The most time-consuming parts of this project were not the model architecture. They were:
- Getting the training pipeline to resume correctly after interruption
- Making the quantization curriculum transitions smooth
- Building the packing/unpacking logic correctly
- Setting up cloud training infrastructure that does not waste money
- Writing documentation that someone else can actually follow

The model architecture is ~500 lines of Python. The infrastructure around it is ~5000 lines. The documentation is ~3000 lines.

This is the nature of systems engineering: the "interesting" part is a small fraction of the total effort. The value is in the complete, working system.
