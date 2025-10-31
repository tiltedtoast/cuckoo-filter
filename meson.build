project(
    'cuckoo-filter-cuda',
    ['cuda', 'cpp'],
    version: '0.1',
    meson_version: '>= 1.3.0',
    default_options: [
        'warning_level=3',
        'cpp_std=c++20',
        'optimization=3',
    ],
)

cuda_args = [
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '--generate-line-info',
    '-arch=sm_80',
]

includes = ['./include']

executable(
    'cuckoo-filter-cuda',
    './src/main.cu',
    include_directories: includes,
    cuda_args: cuda_args,
    install: true,
)

if get_option('BUILD_BENCHMARKS') == true
    benchmark_dep = dependency('benchmark', required: true)
    benchmark_main_dep = dependency('benchmark_main', required: true)
    cpu_cuckoo_dep = dependency('cpu-cuckoo', fallback: ['cpu-cuckoo', 'cpu_cuckoo_dep'])

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

    bench_deps = [benchmark_dep, benchmark_main_dep, cuco_dep, cpu_cuckoo_dep]

    executable(
        'benchmark-cuckoo-vs-bloom',
        './benchmark/cuckoo_vs_bloom.cu',
        dependencies: bench_deps,
        include_directories: includes,
        cuda_args: cuda_args,
        install: false,
    )

    executable(
        'benchmark-cuckoo-cpu-vs-gpu',
        './benchmark/cuckoo_cpu_vs_gpu.cu',
        dependencies: bench_deps,
        include_directories: includes,
        cuda_args: cuda_args,
        install: false,
    )

    executable(
        'benchmark-cuckoo-bucket-size',
        './benchmark/cuckoo_bucket_size.cu',
        dependencies: bench_deps,
        include_directories: includes,
        cuda_args: cuda_args,
        install: false,
    )
endif

if get_option('BUILD_TESTS') == true
    gtest_dep = dependency('gtest', required: true)
    gtest_main_dep = dependency('gtest_main', required: true)

    test_exe = executable(
        'test-cuckoo-filter',
        './tests/test_cuckoo_filter.cu',
        dependencies: [gtest_dep, gtest_main_dep],
        include_directories: includes,
        cuda_args: cuda_args,
        install: false,
    )

    test('cuckoo-filter-tests', test_exe)
endif