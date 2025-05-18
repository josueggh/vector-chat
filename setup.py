"""
Setup script for the vector_chat package.
"""

from setuptools import setup, find_packages

setup(
    name="vector-chat",
    version="0.1.0",
    description="A package for embedding text and creating a chat interface with OpenAI",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "qdrant-client>=1.6.0",
        "nltk>=3.8.1",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "vector-chat=vector_chat.__main__:main",
            "embed=vector_chat.cli.embed:main",
            "chat=vector_chat.cli.chat:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 