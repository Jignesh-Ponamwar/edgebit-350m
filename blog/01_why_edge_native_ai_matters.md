# Why Edge-Native AI Matters

The AI industry is in a paradox. Models are getting more capable every month, but the hardware they require is getting more expensive and concentrated. The largest language models run on clusters that cost millions of dollars. Even "small" models — 7B parameters — need 14GB of RAM just to load the weights in half precision.

Meanwhile, the devices where AI would be most useful — phones, laptops, IoT sensors, medical devices, field equipment — have 2-8GB of RAM, no GPU, and intermittent connectivity.

This is not a temporary problem. It is a structural one.

## The Memory Wall

A 7B-parameter model in FP16 occupies 14GB. A Raspberry Pi 5 has 4GB (or 8GB in the premium model). The model literally does not fit.

Quantization helps: INT4 brings it down to 3.5GB. But that leaves almost no room for the operating system, the KV cache, the application, or anything else. And INT4 quantization is applied after training, which means the model was not designed for it — quality degrades unpredictably.

The approach taken by EdgeBit is different: **design the model for low-bit from the start**.

## Why Not Just Wait for Better Hardware?

Three reasons:

**Privacy.** Data that leaves a device is data that can be intercepted, subpoenaed, or leaked. On-device inference means the data never leaves. For medical, legal, and financial applications, this is not a nice-to-have — it is a requirement.

**Latency.** A round trip to a cloud API takes 200-500ms. On-device inference can produce the first token in under 100ms. For real-time applications (voice assistants, robotics, augmented reality), this difference matters.

**Cost.** Cloud inference costs $0.01-0.10 per request. At scale — millions of devices, continuous inference — this adds up to enormous operating costs. On-device inference has zero marginal cost after deployment.

## What "Edge-Native" Means

An edge-native model is not a compressed version of a cloud model. It is a model designed, from the first training step, to run on constrained hardware.

This means:
- **Architecture-aware quantization**: The model is trained in low-bit precision, not quantized after the fact.
- **Memory-first design**: Every architectural decision (GQA, tied embeddings, INT8 KV cache) targets a memory budget.
- **Runtime engineering**: Weight packing, cache compression, and inference optimization are first-class concerns, not afterthoughts.
- **Deployment completeness**: The project includes everything from training scripts to Docker images to Raspberry Pi deployment guides.

## The EdgeBit Thesis

Useful local AI systems can be built with architecture-aware low-bit training and runtime-first engineering.

This is not about beating GPT-4. It is about making AI work where GPT-4 cannot go: in your pocket, on your desk, at the edge of the network.

EdgeBit-350M is the proof of concept. 334M parameters in 156MB of RAM. It is not the most capable model in the world. But it is designed to be the most deployable.

The value of this project is not in the model quality. It is in the complete, reproducible stack: training, quantization, packing, deployment, benchmarking — all working together as a cohesive system.

That stack, once proven at 350M, can scale.
