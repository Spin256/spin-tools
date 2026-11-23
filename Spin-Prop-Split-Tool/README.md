# Spin Prop Split Tool

> Part of the **Spin Tools** suite: keep this folder inside it. Setup is described in the
> [suite guide](../README.md).

Author: **Spin256** · Website: [spin256.com](https://spin256.com)

Select polygons on one DAZ prop and return the selected pieces and the remainder as
separate static props. UVs, surface assignments and source DAZ materials are kept, and
each piece has its own pivot.

## Install

Run **Install Spin Tools.cmd** and then **Install Spin Tools Menu.dsa**, as described in the
[suite guide](../README.md). Prop Split does not need an NVIDIA GPU; its viewport uses OpenGL.

## First DAZ use

1. Select a prop made of separate geometry islands in DAZ's Scene pane.
2. Open **Scripts → Spin Tools → Split Selected Prop**.
3. Click a polygon, then **Select connected** and **Split off selection**.
4. Optionally rename the piece and place its pivot.
5. Click **Send to DAZ**. The Spin Tools receive window imports the pieces and
   hides the original.

## Use with your own prop

Select one top-level mesh prop or figure with no attached props; for a rigged prop,
select its root above the hip. Start the split action; the controls are on the left.

- **Split off selection** turns the selected polygons into a named piece
  ("Piece 1"…). **Rename piece…** (or double-click) sets the name DAZ uses:
  "<prop> - <name>". Polygons not in a piece form the **Remainder**.
- The **eye icon** beside each piece and the Remainder toggles preview visibility.
  **Ctrl-click** an eye to isolate that piece; **Show all pieces** restores the full
  preview. Hidden faces are excluded from picking but remain in exported pieces.
- **Place pivot on vertex/edge** sets the pivot of the piece chosen in the list,
  including the Remainder. Pieces without one use their bounding-box center.
  A yellow vertex or edge highlight follows the mouse to show exactly where the
  next click will snap. Moving off the mesh or canceling clears the highlight.
- Polygons with more than four sides trigger a warning; **Split n-gons** splits
  them exactly as DAZ does, so the pieces verify on return.
- **Send to DAZ** sends every piece. The original is hidden, never deleted. The
  selector stays open (minimized) so you can adjust and send again; a new send
  hides the previous pieces. Tick **Close after sending** (remembered) to exit
  instead. Closing without sending changes nothing in DAZ.

## Standalone selector and command line

Double-click **Open Spin Prop Split Tool.cmd**, or:

```powershell
& '..\.venv\Scripts\python.exe' '.\app\selector.py' --obj 'C:\path\to\your prop.obj'
```

Standalone mode supports OBJ loading, pieces, pivots and **Export pieces…**:
each piece is written as `<name>-NN-<piece>.obj` beside the chosen file, with a
`<name>.result.json` manifest (names, pivots, source vertex indices). Your source
OBJ is never modified. Options for scripts and other bridges:

```powershell
& '..\.venv\Scripts\python.exe' '.\app\selector.py' --obj 'mesh.obj' `
    --output 'out\mesh.obj' [--result 'out\mesh.result.json'] [--overwrite] `
    [--after-export 'receive.py'] [--close-after on|off] [--allow-ngons]
```

`--after-export` starts `python receive.py <result.json>` once every file is
written (no shell). With a callback the button reads **Send…** and
**Close after send** defaults on; otherwise **Close after export** defaults off.
Both remember your choice; `--close-after` overrides it for one run.
`--allow-ngons` suppresses the n-gon warning and exports n-gons as they are, for
bridges that accept them; the DAZ bridge always needs triangles and quads.

For the advanced manual static-import path, export selection JSON, then run:

```powershell
& '..\.venv\Scripts\python.exe' '.\app\prepare_daz_import.py' '.\my-selection.json'
```

Run **Import Prop Split Selection.dsa** in DAZ and choose the generated `-daz.json`.
The prepared OBJ carries UV/surface names; this manual workflow does not duplicate
the original DAZ shader materials as **Send to DAZ** does.

## Window layout and shortcuts

Every Spin Tools window shares one layout and set of shortcuts; see the
[suite guide](../README.md#shared-window-layout). F1 or ? lists the keys.

## Nested props

Nested props can be selected directly in DAZ. Export bakes evaluated world-space positions (including parent transforms); returned split pieces are independent scene-root props in the same position. Select one mesh without attached prop children; selecting a figure means its figure node above the bones.

## License

See [LICENSE](../LICENSE), [Additional Terms](../ADDITIONAL_TERMS.md) and [Notice](../NOTICE.md).
