{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, self }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = system;
        config.allowUnfree = true;
      };
      cudaPkgs = pkgs.cudaPackages;
      llvm = pkgs.llvmPackages_20;

      cuda = {
        arch = "800";
        sm_target = "sm_80";
        path = cudaPkgs.cudatoolkit;
        version = {
          complete = cudaPkgs.cudaMajorMinorVersion;
          major = cudaPkgs.cudaMajorVersion;
          minor = nixpkgs.lib.lists.last (builtins.splitVersion cuda.version.complete);
        };

      };
    in
    {
      devShells.${system}.default = pkgs.mkShell.override { stdenv = llvm.stdenv; } {
        buildInputs = with cudaPkgs; [
          cudatoolkit
          cuda_cudart
          cuda_cccl
          pkgs.stdenv.cc.cc.lib
        ];

        packages = with pkgs; [
          llvm.clang-tools
          llvm.lldb
          meson
        ];

        CPATH = pkgs.lib.makeIncludePath [
          cudaPkgs.cudatoolkit
        ];

        LD_LIBRARY_PATH =
          pkgs.lib.makeLibraryPath self.devShells.${system}.default.buildInputs + ":/run/opengl-driver/lib";

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
              - -L${cudaPkgs.cudatoolkit}/lib
              - -I${toString ./.}/include
              - -I${toString ./.}/subprojects/cuco/include
              - -D__LIBCUDAXX__STD_VER=${cuda.version.major}
              - -D__CUDACC_VER_MAJOR__=${cuda.version.major}
              - -D__CUDACC_VER_MINOR__=${cuda.version.minor}
              - -D__CUDA_ARCH__=${cuda.arch}
              - --cuda-gpu-arch=${cuda.sm_target}
              - -D__CUDACC_EXTENDED_LAMBDA__
            Remove:
              - -Xcompiler=*
              - -G
              - "-Xfatbin*"
              - "-gencode*"
              - "--generate-code*"
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
