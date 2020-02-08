# Installation

```
$ python -m venv env
Windows: $ env\Scripts\activate
Linux: $ . env/bin/enable
```

```
pip install --editable .
```

After that you can simply invoke `crop` on the terminal.

# Optional dependencies

```
pip install matplotlib
pip install line_profiler
pip install libturbojpeg
Linux: sudo apt install libturbojpeg
```

Execute the line profiler via:
```
kernprof -l -v crop.py
```
