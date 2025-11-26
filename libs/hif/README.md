# HIF (Hypergraph Interchange Format)

Audience: Both

## Overview
HIF is the JSON schema for hypergraph-based explainability artifacts. The schema is versioned and validated with JSON Schema Draft 2020-12.

- Schema: [libs/hif/schema.json](libs/hif/schema.json)
- Validator utilities: [validator.py](libs/hif/validator.py:1)

## Usage example (Python)
```python
from libs.hif.validator import validate_hypergraph, validate_explanation_response

# Minimal hypergraph (example)
hg = {"nodes": [], "hyperedges": []}
validate_hypergraph(hg)  # raises jsonschema.ValidationError on failure

# Minimal explanation envelope
envelope = {"id": "example-1", "status": "completed", "hypergraph": hg}
validate_explanation_response(envelope)
```
