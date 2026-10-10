"""Public branding forwards identical implementations and installable entry points."""
import importlib
import subprocess
import sys
from pathlib import Path

import torchstaar
import staar_phewas

ROOT = Path(__file__).resolve().parents[1]

def test_public_scientific_functions_are_same_objects():
    for name in staar_phewas.__all__:
        assert getattr(torchstaar, name) is getattr(staar_phewas, name)
    for module, names in {
        'chromosome': ('chromosome_configuration', 'run_chromosome', 'main'),
        'cli': ('run_configuration', 'main'),
        'prepare': ('prepare_input', 'read_relationship_matrix', 'main'),
    }.items():
        public = importlib.import_module('torchstaar.' + module)
        original = importlib.import_module('staar_phewas.' + module)
        assert public is not original
        for name in names:
            assert getattr(public, name) is getattr(original, name)
    assert torchstaar.run_chromosome is importlib.import_module('staar_phewas.chromosome').run_chromosome
    assert torchstaar.__version__ == '0.7.0'

def test_installable_metadata_and_only_new_staar_commands():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    from setuptools import find_packages
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert config['project']['name'] == 'torchstaar'
    assert config['project']['version'] == '0.7.0'
    scripts = config['project']['scripts']
    assert not any(name.startswith('staar-phewas') for name in scripts)
    for command in ('torchstaar', 'torchstaar-run'):
        assert scripts[command] == 'staar_phewas.run:main'
        assert callable(importlib.import_module('staar_phewas.run').main)
    for command, module in [('torchstaar-config','cli'), ('torchstaar-chromosome','chromosome'), ('torchstaar-prepare','prepare')]:
        assert scripts[command] == 'torchstaar.' + module + ':main'
        assert callable(importlib.import_module('torchstaar.' + module).main)
    assert scripts['torchstaar-phewas'] == 'torchstaar_phewas.cli:main'
    assert callable(importlib.import_module('torchstaar_phewas.cli').main)
    assert 'torchwgs' not in scripts
    packages = find_packages(str(ROOT), include=config['tool']['setuptools']['packages']['find']['include'])
    assert {'torchstaar', 'staar_phewas', 'torchstaar_phewas'} <= set(packages)
    assert not any(package == 'torchwgs' or package.startswith('torchwgs.') for package in packages)
    assert not (ROOT / 'torchwgs').exists()
    assert 'staar_phewas.cuda_eigen' in packages

def test_three_module_help_entry_points_cpu():
    for module in ('cli', 'chromosome', 'prepare'):
        result = subprocess.run([sys.executable, '-m', 'torchstaar.' + module, '--help'], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert 'usage:' in result.stdout

def test_documented_public_imports_resolve_to_same_implementation_objects():
    import ast
    import re
    paths = [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md'), *(ROOT / 'examples').rglob('*.py')]
    checked = 0
    for path in paths:
        for line in path.read_text().splitlines():
            text = line.strip()
            if not re.match(r'(from torchstaar(?:\.|\s)|import torchstaar(?:\.|\s))', text):
                continue
            node = ast.parse(text).body[0]
            if isinstance(node, ast.ImportFrom):
                public = importlib.import_module(node.module)
                original = importlib.import_module(node.module.replace('torchstaar', 'staar_phewas', 1))
                for item in node.names:
                    if item.name == '*':
                        continue
                    try:
                        actual = getattr(public, item.name)
                    except AttributeError:
                        actual = importlib.import_module(node.module + '.' + item.name)
                    expected = getattr(original, item.name)
                    if isinstance(actual, type(sys)):
                        # Package submodule namespaces are distinct; their objects are shared.
                        for name, value in vars(expected).items():
                            if not name.startswith('_') and callable(value):
                                assert getattr(actual, name) is value
                    else:
                        assert actual is expected, (path, text)
                    checked += 1
            else:
                for item in node.names:
                    importlib.import_module(item.name)
                    checked += 1
    assert checked >= 20

def test_documented_python_module_commands_have_cpu_help():
    import re
    modules = set()
    for path in [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md')]:
        modules.update(re.findall(r'python(?:3)?\s+-m\s+(torchstaar\.[a-z_]+(?:\.[a-z_]+)*)', path.read_text()))
    assert 'torchstaar.cache_runtime.export' in modules
    for module in modules:
        result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (module, result.stderr)
        assert 'usage:' in result.stdout
