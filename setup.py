"""
Package configuration.
"""

from setuptools import find_namespace_packages, setup

setup(
    name="agent-builder",
    version="0.0.1",
    author="Recall Space",
    author_email="info@recall.space",  
    description="The agent-builder package to develop LLM based models.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown", 
    url="https://github.com/Recall-Space/agent-builder",  
    packages=find_namespace_packages(exclude=["tests"]),
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        "langchain==0.3.3",
        "langchain-core==0.3.10",
        "langchain-openai==0.2.2"
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest==8.0.0", "pytest-asyncio==0.23.6"],
    test_suite="tests",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3.10",
    ],
    license="MIT"
)
