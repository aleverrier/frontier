from __future__ import annotations

from setuptools import Extension, setup


setup(
    py_modules=["frontier_native"],
    ext_modules=[
        Extension(
            "_frontier_native",
            sources=["native/_frontier_native.cpp"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
        ),
    ]
)
