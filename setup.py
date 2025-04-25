from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="mean_shift_cy",
        sources=["mean_shift_cy.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++"
    ),
    Extension(
        name="region_growing_cy",
        sources=["region_growing_cy.pyx"],
        include_dirs=[numpy.get_include()],
        language="c++"
    )
]

setup(
    name="segmentation_algorithms",
    ext_modules=cythonize(extensions, annotate=True),
    zip_safe=False,
)