from setuptools import setup, find_packages

setup(
    name="cookbookzombitx64",
    version="0.1.0",
    author="Zombitx64",
    author_email="contact@zombitx64.com",
    description="ไลบรารี่สำหรับการพัฒนาโมเดลเอไอที่ทุกคนเข้าถึงได้ง่าย",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zombitx64/cookbookzombitx64",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
        "torch>=1.9.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "transformers>=4.30.0",
    ],
) 