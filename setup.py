#!/usr/bin/env python
# based on spaCy repository
from setuptools import Extension, setup, find_packages
import sys
import numpy
from setuptools.command.build_ext import build_ext
from sysconfig import get_path
from pathlib import Path
# import shutil
from Cython.Build import cythonize
from Cython.Compiler import Options
import os

ROOT = Path(__file__).parent
PACKAGE_ROOT = ROOT / "src/knowledgator_spacy_vector_matcher"


# Preserve `__doc__` on functions and classes
# http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#compiler-options
Options.docstrings = True

PACKAGES = find_packages()
MOD_NAMES = [
    "src.knowledgator_spacy_vector_matcher.matcher",
    "src.knowledgator_spacy_vector_matcher.utils",
]
COMPILE_OPTIONS = {
    "msvc": ["/Ox", "/EHsc"],
    "mingw32": ["-O2", "-Wno-strict-prototypes", "-Wno-unused-function"],
    "other": ["-O2", "-Wno-strict-prototypes", "-Wno-unused-function"],
}
LINK_OPTIONS = {"msvc": ["-std=c++11"], "mingw32": ["-std=c++11"], "other": []}
COMPILER_DIRECTIVES = {
    "language_level": -3,
    "embedsignature": True,
    "annotation_typing": False,
    "profile": sys.version_info < (3, 12),
}


# By subclassing build_extensions we have the actual compiler that will be used which is really known only after finalize_options
# http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_options:
    def build_options(self):
        for e in self.extensions:
            e.extra_compile_args += COMPILE_OPTIONS.get(
                self.compiler.compiler_type, COMPILE_OPTIONS["other"]
            )
        for e in self.extensions:
            e.extra_link_args += LINK_OPTIONS.get(
                self.compiler.compiler_type, LINK_OPTIONS["other"]
            )


class build_ext_subclass(build_ext, build_ext_options):
    def build_extensions(self):
        if self.parallel is None and os.environ.get("SPACY_NUM_BUILD_JOBS") is not None:
            self.parallel = int(os.environ.get("SPACY_NUM_BUILD_JOBS"))
        build_ext_options.build_options(self)
        build_ext.build_extensions(self)


def clean(path):
    for path in path.glob("**/*"):
        if path.is_file() and path.suffix in (".so", ".cpp", ".html"):
            print(f"Deleting {path.name}")
            path.unlink()


def setup_package():
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        return clean(PACKAGE_ROOT)

    include_dirs = [
        numpy.get_include(),
        get_path("include"),
    ]
    ext_modules = []
    for name in MOD_NAMES:
        mod_path = name.replace(".", "/") + ".pyx"
        ext = Extension(
            name,
            [mod_path],
            language="c++",
            include_dirs=include_dirs,
            extra_compile_args=["-std=c++11"],
        )
        ext_modules.append(ext)
    print("Cythonizing sources")
    ext_modules = cythonize(ext_modules, compiler_directives=COMPILER_DIRECTIVES)

    setup(
        name="knowledgator_spacy_vector_matcher",
        packages=PACKAGES,
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext_subclass},
        package_data={"": ["*.pyx", "*.pxd", "*.pxi"]},
        script_args=["build_ext", "--inplace"]
    )


if __name__ == "__main__":
    setup_package()
