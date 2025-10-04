# Publishing AnalogSim

## Build
```
python -m pip install --upgrade build twine
python -m build
```

## Upload to TestPyPI
```
python -m twine upload --repository testpypi dist/*
```

## Upload to PyPI
```
python -m twine upload dist/*
```

## Versioning
- Start at 0.0.1, then 0.0.2, etc.
- Update CHANGELOG.md for each release.
