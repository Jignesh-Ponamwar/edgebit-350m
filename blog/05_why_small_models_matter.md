# Why Small Models Matter

The AI industry has a scaling obsession. Bigger models, more data, more compute. And it works — GPT-4, Claude, Gemini are remarkable achievements of scale.

But scale creates dependencies. On expensive hardware. On reliable connectivity. On centralized infrastructure. On companies that may change their pricing, terms, or availability at any time.

Small models break these dependencies.

## The Capability Threshold

A 350M parameter model is not going to write a novel or solve differential equations. But it can:

- Classify text into categories
- Extract structured information from documents
- Summarize short passages
- Answer factual questions from provided context
- Follow simple instructions
- Generate template-based responses

These are not impressive demos. They are useful operations that happen millions of times a day in real applications.

A small model that can do these tasks reliably, privately, and cheaply is more valuable in many contexts than a frontier model accessed through an API.

## The Cost Argument

Consider a ticket classification system processing 100,000 tickets per day:

| Approach | Latency | Monthly Cost | Privacy |
|----------|---------|-------------|---------|
| GPT-4 API | 500ms | ~$3,000 | Data sent to OpenAI |
| GPT-3.5 API | 300ms | ~$500 | Data sent to OpenAI |
| Self-hosted 7B | 100ms | ~$1,500 (GPU instance) | Private |
| EdgeBit-350M (CPU) | 50ms | ~$100 (CPU instance) | Private |

The small model is 30x cheaper and keeps data private. For a classification task where 90% accuracy is sufficient, this is the right engineering choice.

## The Latency Argument

Real-time applications need sub-100ms response times:

- Voice assistants: users perceive delays > 200ms
- Autocomplete: must feel instantaneous (< 50ms)
- Content moderation: must process at line rate
- Robotics: control loops run at 10-100Hz

A large model behind an API cannot meet these requirements. A small local model can.

## The Privacy Argument

Some data cannot leave the device:

- Medical records (HIPAA)
- Legal documents (attorney-client privilege)
- Financial transactions (PCI-DSS)
- Personal conversations (user expectation)
- Military/government (classification requirements)

On-device inference is not a preference in these domains. It is a legal requirement.

## The Reliability Argument

Cloud APIs go down. Networks fail. Edge devices operate in environments with intermittent connectivity:

- Agricultural sensors in rural areas
- Industrial equipment in factories
- Mobile devices in tunnels, elevators, airplanes
- Emergency response systems during disasters

A model that runs locally works regardless of connectivity.

## What "Small" Does Not Mean

Small does not mean stupid. A well-trained 350M model with task-specific fine-tuning can outperform a general-purpose 7B model on specific tasks. The key is matching the model to the task.

Small does not mean simple. EdgeBit-350M uses the same architectural innovations as frontier models: GQA, RoPE, SwiGLU, RMSNorm. It is a serious transformer, not a toy.

Small does not mean temporary. The model is not a placeholder until hardware catches up. It is designed for the hardware that exists today and will continue to exist: low-power ARM processors, commodity CPUs, embedded systems.

## The EdgeBit Position

We are not against large models. We use them as teachers (distillation), benchmarks (quality comparison), and inspiration (architectural choices).

But we believe that a complete, deployable small model stack — training, quantization, packing, deployment, benchmarking — is a more useful contribution than another 7B model that requires a GPU.

The AI industry needs both: frontier models that push the boundary of what is possible, and small models that bring AI to the billions of devices where frontier models cannot run.

EdgeBit is a bet on the second path.
