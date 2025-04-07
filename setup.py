"""
Package configuration.
"""

from setuptools import find_namespace_packages, setup

setup(
    name="agent-builder",
    version="0.1.7",
    author="Recall Space",
    author_email="info@recall.space",
    description="The agent-builder package to develop LLM based models.",
    url="https://github.com/Recall-Space/agent-builder",
    packages=find_namespace_packages(exclude=["tests"]),
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        "langchain<1.0.0",
        "langchain-core<1.0.0",
        "langchain-openai<1.0.0",
        "websockets",
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
    license="MIT",
)
