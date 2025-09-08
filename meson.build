project(
    'cuckoo-filter-cuda',
    ['cuda', 'cpp'],
    version: '0.1',
    meson_version: '>= 1.3.0',
    default_options: ['warning_level=3', 'cpp_std=c++20', 'optimization=3'],
)

cuda_args = [
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '-arch=sm_80',
]

cmake = import('cmake')

cuco_opts = cmake.subproject_options()
cuco_opts.add_cmake_defines(
    {
        'BUILD_TESTS': 'OFF',
        'BUILD_BENCHMARKS': 'OFF',
        'BUILD_EXAMPLES': 'OFF',
        'INSTALL_CUCO': 'OFF',
    },
)

cuco_sub = cmake.subproject('cuco', options: cuco_opts)
cuco_dep = cuco_sub.dependency('cuco')

dependencies = [cuco_dep]

includes = ['./include']

executable(
    'cuckoo-filter-cuda',
    './src/main.cu',
    dependencies: dependencies,
    include_directories: includes,
    cuda_args: cuda_args,
    install: true,
)