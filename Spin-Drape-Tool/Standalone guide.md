# Spin Drape Tool — standalone and custom bridges

Double-click **Open Spin Drape Tool.cmd** in the tool folder. This opens a separate
file-loading window. For DAZ, run **Install Spin Tools Menu.dsa** in the Spin Tools
folder once to add the menu actions.
Launched from the .cmd files, diagnostic output (including Warp startup messages) goes to
`temp/startup.log` in the tool folder. If the tool stops while starting, the launcher shows the
end of that log and, for missing or damaged packages, offers to run the installer.

## Loading a scene

- **Props**: static collision meshes, OBJ or FBX.
- **Avatar / moving collision objects**: OBJ for a static object, or FBX for animated
  meshes, including bone skinning and blend shapes. A moving prop belongs here too.
- **Garments**: one or more OBJ or FBX meshes, evaluated in their starting pose.
- **OBJ sequence folders**: one moving object per folder. Files are naturally sorted
  (`frame1`, `frame2`, `frame10`). Keep vertex order and face topology constant.

OBJ inputs must share Y-up world coordinates and the selected units. FBX axes,
transforms and units are converted to Y-up centimeters. All visible polygon meshes
in each FBX are combined for its assigned role; export different roles to separate
files. Hidden meshes are omitted. Geometry must remain consistent over the animation.
The first FBX animation take is used unless a take name is specified.

The sample frame rate controls FBX sampling and OBJ sequence playback. Multiple
moving objects must have the same sample count. Each animation starts at its own
take's beginning and maps to source frame 0 in the helper. There is no timeline
trimming or offset editor. A garment must already align with the
starting collision pose. Static props and garments use the first take's first pose.

Open the scene, run the drape, and pause to export the displayed result. An ordinary
standalone export leaves the window open so you can resume, tug and export again.
Closing the simulation returns to the file-loading window. Closing a directly
launched command-line scene exits the app. No automatic export occurs on cancel.

## Export

Choose **Final static mesh** for OBJ or FBX, or **Animation sequence** for FBX.
Animation output requires moving input and a completed animation run. As in the DAZ
helper, interactive tugging is available for static output. FBX output contains
sampled garment deformation, not the original avatar rig.

OBJ output uses the selected OBJ units. FBX output is Y-up centimeters with unit
metadata. UVs and surface assignments are retained. Existing OBJ material-library
paths are referenced from their original location; textures are not bundled or
converted. FBX shader/texture networks are not imported or reconstructed.

For multiple garments the chosen filename becomes a base name with a numbered
garment suffix. Optional water reference geometry is a separate OBJ, in the selected
OBJ units, even when the garment export is FBX. A detected horizontal Water prop
sets the water height and is excluded from collisions, as in the DAZ workflow.

Every export also writes `name.result.json` **after all meshes have been committed**.
It records absolute output/source paths, labels, vertex counts, units and frame rate.
An export cannot replace an input file. Replacing earlier exports requires confirmation
unless the command line or project file explicitly enables overwrite.

## Command line

From this tool's folder, use the suite's Python (PowerShell example):

```powershell
& '..\.venv\Scripts\python.exe' `
  '.\app\drape_app.py' `
  --prop 'C:\scene\chair.obj' `
  --avatar 'C:\scene\seated-animation.fbx' `
  --garment 'C:\scene\dress.obj' `
  --units cm --fps 30 --output-mode static `
  --output 'C:\scene\draped.obj'
```

Use `--output-mode animation --output C:\scene\draped.fbx` for animation output.
Repeat `--prop`, `--avatar`, `--garment` or `--avatar-sequence` for multiple inputs.
Use `--take NAME` to select an FBX take, `--overwrite` to allow replacing exports,
and `--result PATH` for a manifest name in the same folder as the export.
Command-line inputs automatically open the scene; simulation/export remains interactive.
With no inputs, the file-loading window opens. `drape_app.py --help` lists the options.

## Custom bridge protocol, version 1

Pass `--project bridge.json`. Paths in this JSON are relative to that file:

```json
{
  "version": 1,
  "props": ["chair.obj"],
  "avatars": ["avatar.fbx"],
  "garments": ["dress.obj"],
  "sequences": [],
  "fps": 30,
  "units": "cm",
  "output_mode": "static",
  "output": "returned/dress.obj",
  "result": "returned/dress.result.json",
  "after_export": "receive.py"
}
```

The Python receiver is launched as `python receive.py ABSOLUTE_RESULT_JSON_PATH`.
Use `--after-export SCRIPT.py` for the same behavior without a project JSON.
It runs only after a successful export, with the result folder as its working
directory. Output/errors go to `dress.result.callback.log`. The helper then closes
cleanly, stopping any resident solver worker. It does not wait for the receiving
application to close. A started receiver's later success/failure belongs to that
receiver; launch failures are reported by the helper and exported files remain available.

For other runtimes, replace `after_export` with a **command argument array**:

```json
{
  "callback": ["C:\\MyTool\\receiver.exe", "--drape-result", "{result}"]
}
```

`{result}` is replaced with the absolute JSON path. Use an absolute executable/script
path in this array. No shell is invoked. Do not supply both callback forms. Callbacks
are accepted only through explicit launch options/project configuration; imported
OBJ/FBX files cannot specify a callback. Only use bridge configurations you trust.

Example completed result (abbreviated):

```json
{
  "version": 1,
  "status": "complete",
  "export_id": "unique-export-identifier",
  "output_mode": "animation",
  "coordinates": {"up_axis": "Y", "space": "world", "units": "cm"},
  "fps": 30,
  "start_frame": 0,
  "end_frame": 60,
  "outputs": [{
    "path": "C:\\scene\\returned.fbx",
    "source": "C:\\scene\\dress.obj",
    "label": "dress",
    "vertex_count": 12345
  }]
}
```

A receiver should read only a committed `status: complete` result and use `export_id`
to distinguish new returns from an earlier file. Closing without exporting does not
create a result or invoke the receiver. Session caches use the existing configured
temp folder and are removed after the standalone scene closes. Exported files and
their result JSON remain outside those caches.

## Closing after export or send

The window has a remembered checkbox below the export button:

- **Close after export** (plain standalone, default off): close the simulator once
  the files are written. Off keeps it open to refine and export again.
- **Close after send** (with `--after-export` or a project `callback`, default on):
  exit after the export and its callback start. Off minimizes the window instead,
  so you can restore it, refine and send again; each export runs the callback again.
- **Close after sending** (DAZ, inside Spin Tools, default off): off minimizes
  after each send to the shared receive window.

Each context remembers its own choice. `--close-after on|off` on the command line,
or `"close_after": true|false` in a project JSON, overrides it for that run
without changing the remembered value.
