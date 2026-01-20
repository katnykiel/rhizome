{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python313;
  pythonPackages = python.pkgs;
in
pkgs.mkShell {
  buildInputs = [
    python
    pythonPackages.langchain
    pythonPackages.langchain-ollama
    pythonPackages.numpy
    pythonPackages.scikit-learn
    pythonPackages.pyyaml
    pythonPackages.rich
    pythonPackages.setuptools
    pythonPackages.pip
    pythonPackages.wheel
    pythonPackages.pytest
    pythonPackages.black
    pythonPackages.pylint
  ];

  shellHook = ''
    # Set PYTHONPATH to the rhizome project directory (where this shell.nix is)
    export PYTHONPATH="${toString ./.}:$PYTHONPATH"
    
    # Create alias for rhizome command
    alias rhizome='python -m rhizome.cli'
    
    echo "Rhizome development environment loaded"
    echo "Python version: $(python --version)"
    echo "Project path: ${toString ./.}"
  '';
}
