# Finer

LLM knowledge distillation and fine-tuning tool.

## Installation

```bash
uv sync
```

## Usage

```bash
uv run finer
```

Or run directly:

```bash
uv run python -m src.main
```

## Configuration

Edit `config.json` to customize:
- Training hyperparameters (model, epochs, LoRA settings)
- Data collection settings (models, iterations)
- Logging configuration

## Development

Install dev dependencies:

```bash
uv sync --extra dev
```

Run linting:

```bash
uv run ruff check src/
```

Run type checking:

```bash
uv run pyright
```
