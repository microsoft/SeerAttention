from setuptools import setup, find_packages
import subprocess

subprocess.run(["git", "submodule", "update", "--init", "third_party/cutlass"])

setup(
    name="seer_attn",
    version="0.0.1",
    packages=find_packages(),
)