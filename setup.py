import setuptools

setuptools.setup(
    name="gym_fishing",
    packages=["gym_fishing"],
    version="0.0.5",
    license="MIT",
    description="Provide gym environments for reinforcement learning",
    author="Carl Boettiger & Marcus Lapeyrolerie",
    author_email="cboettig@gmail.com",
    url="https://github.com/boettiger-lab/gym_fishing",
    download_url="https://github.com/boettiger-lab/gym_fishing/archive/v0.0.5.tar.gz",
    keywords=[
        "RL",
        "Reinforcement Learning",
        "Conservation",
        "stable-baselines",
        "OpenAI Gym",
        "AI",
        "Artificial Intelligence",
    ],
    install_requires=["gym", "numpy", "pandas", "matplotlib"],
    extras_require={
        "tests": [
            "stable-baselines3",
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Sort imports
            "isort>=5.0",
            # Reformat
            "black",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",  # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
