from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import platform

extra_compile_args = []
extra_link_args = []

# Flags for Intel i7-10750H (Comet Lake/Skylake derivative)
# CPU supports: SSE4.2, AVX, AVX2, FMA3
# CPU does NOT support: AVX-512

if platform.system() == "Windows":
    # Windows (MSVC) optimizations
    extra_compile_args.extend([
        '/O2',              # Maximize speed
        '/Oi',              # Enable intrinsic functions
        '/Ot',              # Favor fast code
        '/Oy',              # Omit frame pointers
        '/GL',              # Whole program optimization
        '/arch:AVX2',       # AVX2 only (no AVX-512 support!)
        '/fp:fast',         # Fast floating-point
        '/favor:INTEL64',   # Optimize for Intel 64-bit
        '/Gw',              # Optimize global data
        '/GA',              # Optimize for Windows application
        '/Qpar',            # Auto-parallelization hints
        '/openmp',          # OpenMP support
        '/MP12'             # Multi-processor compilation (12 threads)
    ])
    extra_link_args.extend([
        '/LTCG',            # Link-time code generation
        '/OPT:REF',         # Remove unreferenced functions
        '/OPT:ICF'          # Identical COMDAT folding
    ])
else:
    # Linux/MinGW (GCC/Clang) optimizations
    # Use 'skylake' for Comet Lake (Comet Lake is Skylake derivative)
    extra_compile_args.extend([
        '-fopenmp',                 # OpenMP support
        '-O3',                      # Maximum optimization
        '-march=skylake',           # Target Skylake architecture (Comet Lake compatible)
        '-mtune=skylake',           # Tune for Skylake
        '-mavx2',                   # Explicit AVX2 support
        '-mfma',                    # Explicit FMA3 support
        '-ffast-math',              # Fast math optimizations
        '-funroll-loops',           # Loop unrolling
        '-flto',                    # Link-time optimization
        '-fprefer-vector-width=256' # Prefer 256-bit vectors (AVX2)
    ])
    extra_link_args.extend([
        '-fopenmp',
        '-flto'
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
    version="2.0.0",
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