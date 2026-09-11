"""Build the optional C fast path next to the pure-Python reference.

The package stays importable without the extension: every caller keeps the
Python implementation as a fallback. Build in place with
`python setup.py build_ext --inplace`.
"""

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "mahjong_ai._fast",
            sources=["src/mahjong_ai/_fast.c"],
            extra_compile_args=["-O3"],
        )
    ]
)
