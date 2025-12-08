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

cli11_dep = subproject('cli11').get_variable('CLI11_dep')

includes = ['./include']
deps = [cli11_dep]

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

nccl_dep = dependency('nccl', required: false)

if nccl_dep.found()
    executable(
        'cuckoo-filter-multi-gpu',
        './src/multi_gpu_main.cu',
        include_directories: includes,
        dependencies: deps + [nccl_dep],
        cuda_args: cuda_args,
        install: true,
    )
endif

if get_option('BUILD_BENCHMARKS') == true
    benchmark_dep = subproject('google-benchmark').get_variable('google_benchmark_dep')
    cpu_cuckoo_dep = subproject('cpu-cuckoo').get_variable('cpu_cuckoo_dep')

    cmake = import('cmake')

    cuco_opts = cmake.subproject_options()
    cuco_opts.add_cmake_defines(
        {
            'CMAKE_BUILD_TYPE': 'Release',
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
            'CMAKE_BUILD_TYPE': 'Release',
            'BUILD_TESTING': 'OFF',
        },
    )
    partitioned_sub = cmake.subproject('partitioned-filters', options: partitioned_opts)
    partitioned_dep = partitioned_sub.dependency('filters')

    gqf_subproj = subproject('gqf')
    gqf_dep = gqf_subproj.get_variable('gqf_dep')
    gqf_dep_8 = gqf_subproj.get_variable('gqf_deps')['8']
    gqf_dep_16 = gqf_subproj.get_variable('gqf_deps')['16']
    gqf_dep_32 = gqf_subproj.get_variable('gqf_deps')['32']

    tcf_opts = cmake.subproject_options()
    tcf_opts.add_cmake_defines(
        {
            'CMAKE_BUILD_TYPE': 'Release',
            'BUILD_TESTING': 'OFF',
        },
    )
    tcf_sub = cmake.subproject('tcf', options: tcf_opts)
    tcf_dep = tcf_sub.dependency('bulk_tcf')

    bench_deps = [
        benchmark_dep,
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
        'benchmark-cuckoo-sorted-vs-unsorted',
        './benchmark/cuckoo_sorted_vs_unsorted.cu',
        dependencies: bench_deps,
        include_directories: includes,
        cuda_args: cuda_args,
        install: false,
    )

    executable(
        'benchmark-eviction',
        './benchmark/eviction_benchmark.cu',
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

    executable(
        'benchmark-cuckoo-vs-tcf',
        './benchmark/cuckoo_vs_tcf.cu',
        dependencies: bench_deps + [tcf_dep],
        include_directories: includes,
        cuda_args: cuda_args + ['-w'],
        install: false,
    )

    executable(
        'benchmark-load-factor',
        './benchmark/load_factor_benchmark.cu',
        dependencies: bench_deps + [cuco_dep, tcf_dep, partitioned_dep, cpu_cuckoo_dep, gqf_dep],
        include_directories: includes,
        cuda_args: cuda_args + ['-w'],
        install: false,
    )

    executable(
        'benchmark-fpr',
        './benchmark/fpr_benchmark.cu',
        dependencies: bench_deps + [cuco_dep, tcf_dep, partitioned_dep, cpu_cuckoo_dep, gqf_dep],
        include_directories: includes,
        cuda_args: cuda_args + ['-w'],
        install: false,
    )

    executable(
        'benchmark-fpr-sweep-gqf8',
        './benchmark/fpr_sweep_benchmark.cu',
        dependencies: bench_deps + [cuco_dep, tcf_dep, gqf_dep_8],
        include_directories: includes,
        cuda_args: cuda_args + ['-w', '-DGQF_BITS=8'],
        install: false,
    )

    executable(
        'benchmark-fpr-sweep-gqf16',
        './benchmark/fpr_sweep_benchmark.cu',
        dependencies: bench_deps + [cuco_dep, tcf_dep, gqf_dep_16],
        include_directories: includes,
        cuda_args: cuda_args + ['-w', '-DGQF_BITS=16'],
        install: false,
    )

    executable(
        'benchmark-fpr-sweep-gqf32',
        './benchmark/fpr_sweep_benchmark.cu',
        dependencies: bench_deps + [cuco_dep, tcf_dep, gqf_dep_32],
        include_directories: includes,
        cuda_args: cuda_args + ['-w', '-DGQF_BITS=32'],
        install: false,
    )

    executable(
        'profiler-benchmark',
        './benchmark/profiler_benchmark.cu',
        dependencies: [cli11_dep, cuco_dep, tcf_dep, benchmark_dep, gqf_dep],
        include_directories: includes,
        cuda_args: cuda_args,
        install: false,
    )


    if nccl_dep.found()
        executable(
            'benchmark-cuckoo-single-vs-multi-gpu',
            './benchmark/cuckoo_single_vs_multi_gpu.cu',
            dependencies: bench_deps + [nccl_dep],
            include_directories: includes,
            cuda_args: cuda_args,
            install: false,
        )
    endif
endif

if get_option('BUILD_TESTS') == true
    gtest_dep = subproject('gtest').get_variable('gtest_dep')
    gtest_main_dep = subproject('gtest').get_variable('gtest_main_dep')

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