from __future__ import annotations

from setuptools import Extension, setup


setup(
    ext_modules=[
        Extension(
            "_frontier_fast_native",
            sources=["native/_frontier_fast_native.cpp"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
        ),
    ]
)
