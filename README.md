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

## Turbo JPEG

This will speedup the jpeg processing by 2-4x. First install the python package.

```
pip install libturbojpeg
```

Then install the library `libturbojpeg`.
+ Linux: Install via `sudo apt install libturbojpeg`
+ Windows: Installer `libjpeg-turbo-*-vc64.exe` from https://sourceforge.net/projects/libjpeg-turbo/files
```

## Matplotlib

This package is required for the debug plots.

```
pip install matplotlib
```

Now you can invoke the `crop``  tool with the `--debug` option. Then you should see some plots.


## Line Profiler

Profile the source code. First install the package.

```
pip install line_profiler
```

Then start the profiler via:
```
kernprof -l -v crop ..
```

# Useful scripts

To reverse the order of images, invoke the following commands in a python shell.

```
import os
names = list(sorted(os.listdir(".")))
print(names)
for i, name in enumerate(reversed(names)):
	os.rename(name, "{:03d}.JPG".format(i))
```
