from setuptools import setup, find_packages

setup(
    name="memory_system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "openai",
        "numpy",
        "faiss-cpu",  # 或 faiss-gpu
        "ollama"
    ],
)