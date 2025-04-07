# 🧠 Practical deep dive into Adam and AdamW

This project accompanies the blog post on [Adam Vs. AdamW](https://adelbennaceur.github.io/posts/adam_vs_adamw/) and features NumPy and PyToch implementations to help better understand the concepts.

## 🚀 How to Run

Make sure uv is installed (see [installation](https://docs.astral.sh/uv/getting-started/installation/)) and then run:
To compare the optimizers:

```bash
uv run adam_vs_adamw_verification.py
```

This will generate the loss curves and decision boundaries figures in the `assets` directory.

To run the numpy implementation:

```bash
uv run numpy_implementations.py
```

To run the Pytorch implementation:

```bash
uv run pytorch_implementation.py
```

Using the command `uv run` will take care of the venv and installing dependencies.

## 📝 License

MIT License
