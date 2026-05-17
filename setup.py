from setuptools import setup, find_packages

setup(
    name="edgebit",
    version="0.1.0",
    description="EdgeBit-350M: 1.58-bit edge-native transformer",
    author="Jignesh Ponamwar",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "safetensors>=0.4.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "train": [
            "accelerate>=0.25.0",
            "datasets>=2.16.0",
            "deepspeed>=0.12.0",
        ],
        "eval": [
            "lm-eval>=0.4.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "edgebit-train=training.train:main",
            "edgebit-bench=runtime.bench_runtime:main",
            "edgebit-export-hf=export.export_hf:main",
            "edgebit-export-gguf=export.export_gguf:main",
            "edgebit-export-packed=export.export_packed:main",
            "edgebit-eval=eval.run_lm_eval:main",
        ],
    },
)
