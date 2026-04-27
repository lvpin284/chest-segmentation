from setuptools import setup, find_packages

setup(
    name="pleural-effusion-agent",
    version="0.1.0",
    description="超声胸腔积液智能分析Agent / Ultrasound Pleural Effusion Intelligent Analysis Agent",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "Pillow>=9.5.0",
        "opencv-python-headless>=4.7.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "matplotlib>=3.7.0",
    ],
)
