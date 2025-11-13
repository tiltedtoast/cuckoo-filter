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
    '-gencode=arch=compute_120,code=sm_120',
    '-gencode=arch=compute_75,code=compute_75',
]

includes = ['./include', './benchmark']
deps = [
    dependency('CLI11', required: true),
]

executable(
    'cuckoo-filter-ipc',
    './src/ipc_main.cu',
    include_directories: includes,
    dependencies: deps,
    cuda_args: cuda_args,
    install: true,
)

executable(
    'cuckoo-filter-ipc-server',
    './src/ipc_server_main.cu',
    include_directories: includes,
    dependencies: deps,
    cuda_args: cuda_args,
    install: true,
)

executable(
    'cuckoo-filter-cuda',
    './src/main.cu',
    include_directories: includes,
    dependencies: deps,
    cuda_args: cuda_args,
    install: true,
)

executable(
    'cuckoo-filter-multi-gpu',
    './src/multi_gpu_main.cu',
    include_directories: includes,
    dependencies: deps,
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

    partitioned_opts = cmake.subproject_options()
    partitioned_opts.add_cmake_defines(
        {
            'BUILD_TESTING': 'OFF',
        },
    )
    partitioned_sub = cmake.subproject('partitioned-filters', options: partitioned_opts)
    partitioned_dep = partitioned_sub.dependency('filters')

    gqf_dep = dependency('quotient-filters', fallback: ['quotient-filters', 'gqf_dep'])

    bench_deps = [
        benchmark_dep,
        benchmark_main_dep,
        cpu_cuckoo_dep,
    ]

    executable(
        'benchmark-cuckoo-vs-bloom',
        './benchmark/cuckoo_vs_bloom.cu',
        dependencies: bench_deps + [cuco_dep],
        include_directories: includes,
        cuda_args: cuda_args,
        install: false,
    )

    executable(
        'benchmark-cuckoo-cpu-vs-gpu',
        './benchmark/cuckoo_cpu_vs_gpu.cu',
        dependencies: bench_deps + [cpu_cuckoo_dep],
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

    executable(
        'benchmark-cuckoo-ipc-overhead',
        './benchmark/cuckoo_ipc_overhead.cu',
        dependencies: bench_deps,
        include_directories: includes,
        cuda_args: cuda_args,
        install: false,
    )

    executable(
        'benchmark-cuckoo-vs-partitioned',
        './benchmark/cuckoo_vs_partitioned.cu',
        dependencies: bench_deps + [partitioned_dep],
        include_directories: includes,
        cuda_args: cuda_args + ['-w'],
        install: false,
    )

    executable(
        'benchmark-cuckoo-vs-gqf',
        './benchmark/cuckoo_vs_gqf.cu',
        dependencies: bench_deps + [gqf_dep],
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
