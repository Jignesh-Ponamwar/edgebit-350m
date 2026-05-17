# Data Directory

Place training data here in JSONL format.

## Expected Format

### Pretraining Data
```json
{"text": "The quick brown fox jumps over the lazy dog."}
{"text": "Machine learning enables automated pattern recognition."}
```

### Instruction Data
```json
{"instruction": "Summarize the following text.", "response": "The text describes..."}
```

Or pre-formatted:
```json
{"text": "### Instruction:\nSummarize the text.\n\n### Response:\nThe text describes..."}
```

## Generating Synthetic Data

For smoke testing without real data:
```bash
python -c "from training.data import create_synthetic_data; create_synthetic_data('data/synthetic_train.jsonl', n_samples=5000)"
```

## Recommended Datasets

For actual training, consider these open datasets:

| Dataset | Size | Description |
|---------|------|-------------|
| SlimPajama | 627B tokens | Deduplicated RedPajama |
| Dolma | 3T tokens | AI2 open pretraining corpus |
| The Pile (subset) | 825GB | Diverse text corpus |
| OpenAssistant | 161K conversations | Instruction following |
| Alpaca | 52K instructions | Instruction following |

Download and convert to JSONL format before training.
