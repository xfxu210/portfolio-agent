# Model Providers

Execution mode and provider identity are separate:

- `mock`: deterministic local contract implementation
- `live`: selected external provider with explicit model ID
- `recorded`: immutable playback of genuine captured provider output

| Provider | Environment variable | Transport |
|---|---|---|
| Muse | `MODEL_API_KEY` | Meta Model API at `https://api.meta.ai/v1` |
| OpenAI | `OPENAI_API_KEY` | Official OpenAI SDK |
| Anthropic | `ANTHROPIC_API_KEY` | Official Anthropic SDK |
| Gemini | `GOOGLE_API_KEY` | Official Google GenAI SDK |

Keys remain backend-only. Missing credentials fail before dataset processing. Vendor responses
are normalized through identical Pydantic schemas. Refusals, empty output, network failures,
and schema-invalid output are explicit provider errors; the system never silently swaps a
provider or substitutes mock output.

Model APIs and supported structured-output parameters change. Validate model IDs and current
provider capabilities before a live run. Exact provider responses—not presumed sampling
determinism—are the basis of recorded replay.

