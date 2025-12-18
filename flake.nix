{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-compat = {
      url = "github:NixOS/flake-compat";
      flake = false;
    };
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = system;
        config.allowUnfree = true;
      };

      # https://github.com/clangd/clangd/issues/2531
      cudaPkgs = pkgs.cudaPackages_12_9;
      llvm = pkgs.llvmPackages_21;

      cuda = {
        arch = "1200";
        sm_target = "sm_120";
        path = cudaPkgs.cudatoolkit;
        version = {
          complete = cudaPkgs.cudaMajorMinorVersion;
          major = cudaPkgs.cudaMajorVersion;
          minor = nixpkgs.lib.lists.last (builtins.splitVersion cuda.version.complete);
        };

      };
      buildInputs = with cudaPkgs; [
        cudatoolkit
        cuda_cudart
        pkgs.stdenv.cc.cc.lib
      ];

      nativeBuildInputs = with pkgs; [
        llvm.clang-tools
        llvm.lldb
        meson
        uv
        pkg-config
        doxygen
        graphviz
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {

        inherit buildInputs nativeBuildInputs;

        CPATH = pkgs.lib.makeIncludePath [
          cudaPkgs.cudatoolkit
        ];

        LD_LIBRARY_PATH = "${
          pkgs.lib.makeLibraryPath (buildInputs ++ nativeBuildInputs)
        }:/run/opengl-driver/lib";

        shellHook = ''
              if [ ! -e .clangd ]; then
                cat > .clangd <<EOF
          CompileFlags:
            Compiler: ${cudaPkgs.cudatoolkit}/bin/nvcc
            Add:
              - -xcuda
              - --cuda-path=${cudaPkgs.cudatoolkit}
              - -D__INTELLISENSE__
              - -D__CLANGD__
              - -I${cudaPkgs.cudatoolkit}/include
              - -I$(pwd)/include
              - -I$(pwd)/subprojects/cuco/include
              - -I$(pwd)/subprojects/googletest-1.17.0/googletest/include
              - -I$(pwd)/subprojects/openssl-3.0.8/include
              - -I$(pwd)/subprojects/gossip/include
              - -D__LIBCUDAXX__STD_VER=${cuda.version.major}
              - -D__CUDACC_VER_MAJOR__=${cuda.version.major}
              - -D__CUDACC_VER_MINOR__=${cuda.version.minor}
              - -D__CUDA_ARCH__=${cuda.arch}
              - --cuda-gpu-arch=${cuda.sm_target}
              - -D__CUDACC_EXTENDED_LAMBDA__
            Remove:
              - -Xcompiler=*
              - -G
              - "-arch=*"
              - "-Xfatbin*"
              - "-gencode*"
              - "--generate-code*"
              - "--generate-line-info"
              - "--compiler-options*"
              - "--expt-extended-lambda"
              - "--expt-relaxed-constexpr"
              - "-forward-unknown-to-host-compiler"
              - "-Werror=cross-execution-space-call"

          Diagnostics:
            UnusedIncludes: None
            Suppress:
              - variadic_device_fn
              - attributes_not_allowed
              - undeclared_var_use_suggest
              - typename_invalid_functionspec
              - expected_expression
          EOF
                echo ".clangd created by flake shellHook"
              fi
        '';
      };
    };
}
