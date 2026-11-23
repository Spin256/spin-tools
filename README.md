# Spin Tools

Author: **Spin256** · [spin256.com/tools](https://www.spin256.com/tools)

Six tools, installed together. The five helper apps share one Python environment:

- **Spin Drape Tool** — cloth simulation, standalone OBJ/FBX import/export and DAZ bridge.
- **Spin Soft Prop Tool** — cushions, mattresses and other soft props under prescribed
  static or animated contact, standalone import/export and DAZ bridge.
- **Spin Soft Body Tool** — painted static/animated contact deformation and jiggle
  for figures, with rigid ghost regions and preview opacity by category.
- **Spin Prop Split Tool** — select polygons on a DAZ prop and return named, separate pieces.
- **Spin Hair Tool** — settle DAZ card hair against the avatar on the GPU, keeping the style,
  with freezing for ties and buns, gravity, wind, water and live pulling, for stills and animation;
  standalone OBJ/FBX import/export and DAZ bridge.
- **Current Frame to Zero** — keep the scene's current keyed state at frame 0 and
  remove the other animation keys. This DAZ-only utility needs no Python.

All six appear under **Scripts > Spin Tools** in DAZ Studio.

## Install and open

1. Install 64-bit Windows Python 3.10–3.13 separately. Python 3.10 is tested.
2. Download the complete Spin-Tools folder and run **Install Spin Tools.cmd**.
   This creates the suite's `.venv`, installs the pinned requirements and checks
   all five helper apps. Python itself is not bundled. First installation needs internet
   access unless an offline wheel folder is provided.
3. Drag **Install Spin Tools Menu.dsa** from File Explorer into DAZ Studio's viewport.
   It adds every tool under **Scripts > Spin Tools** and shows one summary. Matching
   existing entries are updated to this folder's location instead of duplicated. This is
   the only menu installer you need to run.

DAZ Studio and NVIDIA drivers are supplied separately. Drape, Soft Prop, Soft Body and Hair require
an NVIDIA CUDA GPU; Prop Split uses VTK/OpenGL and does not itself use Warp.

For standalone use (without DAZ), double-click **Open Spin Tools.cmd**: it shows a button for each
tool. Each tool folder also has its own **Open … .cmd**, which takes command-line options.
If Spin Tools is not installed yet, or a tool stops while starting because packages are
missing or damaged, the launcher offers to run **Install Spin Tools.cmd**, which repairs it.
The launcher also shows how much space the temporary files take (captured scenes, results,
animation frames and logs in each tool's `temp` folder) and **Clear temporary files…** deletes
them once every tool is closed. Import any result waiting in DAZ first.
Drape, Soft Prop and Soft Body import/export OBJ and FBX, including baked mesh
animation where supported. Prop Split opens OBJ files and
exports each piece as OBJ. Hair opens card hair and an avatar (OBJ, animated FBX or OBJ
frames) and exports OBJ or FBX.

The install commands inside the tool folders set up this same shared environment.
Keep the whole suite together. After moving it, run **Install Spin Tools.cmd** again and
then **Install Spin Tools Menu.dsa**, so DAZ points at the new location. Do not copy an
installed `.venv` to another computer; run setup there instead.

### Setup options

Setup looks for Python in the suite's `.venv`, then the Python launcher (`py`), then PATH.
To choose one, run setup with it from this folder:

```text
"C:\Path\To\Python310\python.exe" app\setup_environment.py
```

To install without internet access, download the packages on a connected PC with the same
Windows and Python version, copy the folder over, then pass it to the installer:

```text
py -3.10 -m pip download --only-binary=:all: -r requirements-lock.txt -d wheelhouse
"Install Spin Tools.cmd" --wheelhouse .\wheelhouse
```

Setup writes its progress and errors to `temp/install.log`. If it fails, the tools'
settings are left as they were; run it again to repair missing packages. Setup never
deletes a `.venv` it does not recognize.

To check an installation at any time (package versions, the GPU and that every tool
loads; setup runs this at the end):

```text
.venv\Scripts\python.exe app\check_install.py
```

## Tool guides

- [Drape guide](Spin-Drape-Tool/README.md)
- [Soft Prop guide](Spin-Soft-Prop-Tool/README.md)
- [Soft Body guide](Spin-Soft-Body-Tool/README.md)
- [Hair guide](Spin-Hair-Tool/README.md)
- [Prop Split guide](Spin-Prop-Split-Tool/README.md)
- [Current Frame to Zero guide](Spin-Frame-Zero-Tool/README.md) for collapsing a
  scene animation independently of the simulation tools.

## Folder layout

```text
Spin-Tools/
  Install Spin Tools.cmd
  Install Spin Tools Menu.dsa
  Open Spin Tools.cmd        standalone launcher (a button per tool)
  keybindings.json           keyboard shortcuts
  requirements-lock.txt
  .venv/                     shared installed dependencies
  app/                       suite setup, launcher and health checks
  temp/                      suite installation logs/cache
  Spin-Drape-Tool/            cloth draping
  Spin-Soft-Prop-Tool/        soft props
  Spin-Soft-Body-Tool/        soft body surface deformation
  Spin-Prop-Split-Tool/       prop splitting
  Spin-Hair-Tool/             card-hair strands, GPU settling and freezing
  Spin-Frame-Zero-Tool/       independent DAZ frame utility
```

Setup creates `.venv` and `temp`. Each tool's `settings.json` holds its session folder
(`session_folder`, default `temp`) and its Python (`python_executable`, default
`../.venv/Scripts/pythonw.exe`). Both paths are relative to that tool's folder unless
absolute. Do not remove session files while a tool or a DAZ return is active.

## Keyboard shortcuts

All tools read `keybindings.json` in this folder: a `common` section shared by every tool
and a section per tool. Edit it and reopen the tool.

## Shared window layout
All Spin Tools helper windows (Drape, Hair, Soft Body, Soft Prop, Prop Split) share one layout:
collapsible sections on the left (open/closed state remembered), the same button row below them
(Run/Pause/Resume, Reset, Send to DAZ, Close after sending), the Spin256 footer, and a status bar
under the viewport: simulation state on the left, frame x of y during timeline work in the
middle, the current activity (with a progress bar where the length is known) on the right.
The View section shows/hides the simulated object and the avatar/colliders and sets their
opacity. Keys come from the suite's keybindings.json (W/A/S/D orbit, arrows pan, PgUp/PgDown
zoom, V reset view, Space run/pause, Esc cancel, Ctrl+Return send); press F1 or ? in any tool
for the list. A tool opened for the first time uses the size and position of the last Spin tool
you closed; after that it remembers its own.

## License

Spin Tools is licensed under AGPL-3.0-only: see [LICENSE](LICENSE) and
[Additional Terms](ADDITIONAL_TERMS.md). [Notice](NOTICE.md) explains the intent behind it.
Installed packages keep their own licenses in `.venv`;
[NVIDIA Warp](https://github.com/NVIDIA/warp#license) is Apache-2.0.
