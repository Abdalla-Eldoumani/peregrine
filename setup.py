from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import platform

extra_compile_args = []
extra_link_args = []

# CPU-specific optimization flags for Windows
cpu_specific_flags = [
    '/arch:AVX2',
    '/favor:INTEL64',
    '/GA',
    '/O2',
    '/Oi',
    '/Ot',
    '/Oy',
    '/fp:fast',
    '/GL',
    '/Gw',
    '/openmp',
    '/openmp:experimental',
    '/MP12'
]

if platform.system() == "Windows":
    extra_compile_args.extend(cpu_specific_flags)
    extra_compile_args.extend([
        '/openmp:experimental',
        '/Qopenmp',
        '/arch:AVX512',
        '/fp:fast',  
        '/Qopenmp',  
        '/Qopenmp-simd',
        '/Qpar',
        '/O2', '/arch:AVX2', '/fp:fast', '/GL'
    ])
    extra_link_args.extend([
        '/LTCG',     # Link-time Code Generation
        '/OPT:REF',  # Eliminate Unreferenced Data
        '/OPT:ICF',  # Identical COMDAT Folding
    ])
    extra_link_args.extend([
        '/LTCG',     # Link-time Code Generation
        '/OPT:REF',  # Eliminate Unreferenced Data
        '/OPT:ICF',  # Identical COMDAT Folding
    ])
else:
    extra_compile_args.extend([
        '-fopenmp',
        '-O3',
        '-march=comet-lake',
        '-mtune=comet-lake',
        '-mavx2',
        '-mfma',
        '-ffast-math',
        '-funroll-loops',
        '-floop-optimize',
        '-flto',
        '-fopt-info-vec',
        '-fopt-info-vec-missed',
        '-fprefer-vector-width=256',
        '-fopenmp-simd',
        '-mavx512f',
        '-mavx512cd'
    ])
    extra_link_args.extend([
        '-fopenmp',
        '-flto',
    ])

ext_modules = [
    Pybind11Extension(
        "MathExt",
        ["MathExt.cpp"],
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[
            ('CPU_INTEL_COMET_LAKE', '1'),
            ('AVX2_AVAILABLE', '1'),
            ('FMA_AVAILABLE', '1'),
            ('NUM_CPU_CORES', '6'),
            ('NUM_CPU_THREADS', '12'),
        ],
    ),
]

setup(
    name="MathExt",
    version="1.0.0",
    author="Abdalla ElDoumani",
    description="High-performance mathematical operations using C++ and SIMD",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.6",
    install_requires=['pybind11>=2.6.0'],
    setup_requires=['pybind11>=2.6.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)