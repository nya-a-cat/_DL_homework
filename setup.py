import setuptools
import os

# 读取 README.md 作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取 requirements.txt 获取依赖
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(req_path, 'r', encoding='utf-8') as f:
        # 过滤掉注释和空行
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements

setuptools.setup(
    name="my_nn_lib", # 包名，pip install my_nn_lib 时使用
    version="0.0.1", # 初始版本号
    author="Catneko", # 替换成你的名字
    author_email="your.email@example.com", # 替换成你的邮箱
    description="一个正在开发中的神经网络库(为了作业)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/your_repo_name", # 替换成你的项目仓库 URL (可选)
    project_urls={ # 可选的额外链接
        "Bug Tracker": "https://github.com/your_username/your_repo_name/issues",
    },
    classifiers=[ # PyPI 分类标签 (https://pypi.org/classifiers/)
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # 假设使用 MIT 许可证
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha", # 开发状态
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "."}, # 告诉 setuptools 包在根目录下
    packages=setuptools.find_packages(where="."), # 自动查找 my_nn_lib 及其子包
    python_requires=">=3.8", # 指定支持的 Python 版本
    install_requires=read_requirements(), # 从 requirements.txt 读取依赖
    # 如果有 C++/CUDA 扩展，后续需要在这里添加 ext_modules
    # ext_modules=[...],
    # cmdclass={'build_ext': ...}, # 用于自定义编译过程
)