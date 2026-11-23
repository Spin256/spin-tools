# Spin Soft Prop Tool

> Part of the **Spin Tools** suite: keep this folder inside it. Setup is described in the
> [suite guide](../README.md).

Deform a cushion, mattress or other solid soft prop under a posed or moving object.
Author: **Spin256** · [spin256.com](https://spin256.com)

The collision object follows its supplied pose or animation. Only the soft prop
is simulated: a bouncing sphere can dent a mattress, but the mattress does not
change the sphere's movement.

## Install

Run **Install Spin Tools.cmd** and then **Install Spin Tools Menu.dsa**, as described in the
[suite guide](../README.md). An NVIDIA CUDA GPU is required.

## DAZ Studio

Use **Scripts > Spin Tools > Spin Soft Prop Tool**:

1. Assign collision objects, such as an avatar or a sphere.
2. Assign the soft prop, such as a cushion. Each assignment includes only that
   object's own mesh, so assign any additional child meshes explicitly.
3. Review the objects and choose current-pose static output or timeline animation.
4. In the helper, choose material/support settings and run. Pause a static run to
   return what is displayed, or wait for the complete animation, then send to DAZ.

The returned prop keeps its vertex order, UVs, surfaces and materials.

## Standalone

The launcher opens a file selector with two lists: **Collision objects** and
**Soft props to deform**. Add OBJ or FBX files and choose the output mode.
Static mode uses a single pose. Animation mode follows a collision object's FBX
animation or an OBJ sequence folder. All animated inputs must have matching sample
counts. OBJ frames must keep the same vertex order and triangulated topology.

OBJ coordinates are world-space, Y-up; choose their units. FBX axes, world
transforms and units are converted during import. The soft prop must start in
the correct position. Imported soft-prop animation is not used.

Static results export to OBJ or FBX; animation results export to FBX. An export
manifest is written next to the output. Multiple soft props produce separate
files. Input files cannot be overwritten. Material/texture references are retained
where available; exports do not package external texture assets.

## Controls

- **Shape firmness** resists stretching and distortion.
- **Compression firmness** controls resistance to loss of volume independently of
  the surface tension. Very large prescribed overlaps
  can still exceed the model's limits.
- **Surface tension** spreads dents into neighboring areas and smooths sudden
  changes in deformation. For example, the cushion between two legs is pulled
  down by both contact areas. It smooths displacement rather than the original
  mesh, preserving the unloaded shape. Zero disables it. Start at 35%; try 60%
  for a more connected depression. Strong tension can reduce bounce and local detail.
- **Contact clearance** sets the requested separation in millimeters, default
  0.2 mm. Zero removes the deliberate gap. The coarse cage can still leave local
  differences from that requested clearance.
- **Damping** removes motion; lower values allow more bounce and overshoot.
- **Shape recovery** pulls toward the original shape. Zero removes this extra
  pull; the elastic cage still resists deformation.
- **Density** affects inertia. Presets are artistic starting points, not measured
  material specifications.
- **Hold underside** anchors the lowest cage layer of each prop. Paint additional
  fixed supports with the left mouse button, or select erase. Supports are fixed
  in world space and affect neighboring cage nodes as well as painted vertices.
- **Static settling** controls each run/resume batch. The result pauses at its end;
  Resume continues the existing simulation. Reset restores the source shape.
- **Initial pose settling** holds the first animation pose briefly before advancing
  through the supplied frames. Animation output always contains the full sequence;
  a partial sequence cannot be exported.
- **Cage resolution** adds internal detail and cost; solver iterations improve
  constraint convergence within each simulation step.

Material/support changes require a new run. Inputs are disabled during simulation.
For a stool with a separate rigid stand, simulate the cushion separately and keep
the stand rigid. This gives the cushion its own cage instead of spanning the legs
and underside of the whole stool.
Space runs/pauses/resumes, V resets the view, Ctrl+Enter exports an available result.
With painting disabled, the mouse uses the normal trackball view controls.
Window size and the sidebar width are remembered.

## Seamless loop

**Loop animation (carry motion)** makes a timeline animation repeat seamlessly. A
simulation starts from rest on its first frame, so a plain loop restarts the motion at the join. With this
on, the simulation keeps going past the last frame into frames 1, 2, ... on the same avatar poses until it
meets the first run (within 0.2 mm), and the first run is used from there; frame 0 is that simulated step.
If they have not met by the middle of the animation, the closest frame is used and the second pass is steered
onto it over **Loop landing frames** (default 8). The finished status says where they met. The avatar
animation's last pose should be the one *before* its first: after it the simulation takes one real step into frame 0 (a repeated first frame just holds for one frame). It costs up to half the animation again in simulation time.

## Other application bridges and command line

The `.cmd` launcher forwards arguments to the app. For example:

```text
"Open Spin Soft Prop Tool.cmd" --collider "sphere.obj" --soft-prop "cushion.obj" --units cm
"Open Spin Soft Prop Tool.cmd" --project "my-project.json"
```

A version 1 project JSON accepts `colliders`, `soft_props`, `collider_sequences`
(arrays of file/folder paths), `units` (`cm`, `m`, `mm`), `fps`, `take`,
`output_mode` (`static`, `animation`), `output`, `result`, and `overwrite`.
Paths in JSON are relative to that JSON file. Command-line paths are relative
to the launcher's working folder. The file selector remains available before simulation.

For a custom bridge, add `after_export` pointing to a Python script. The helper
runs it with this tool's Python and the completed result-manifest path as its first
argument. Alternatively, use a `callback` argument array with a `{result}` placeholder
for your application's receiver executable. No shell is used. Choose one callback
mechanism. The receiver is responsible for importing the result into its application.
Successful callback exports close the helper after its simulation worker stops.
Callbacks are configured explicitly; ordinary standalone exports keep the helper open.

The command-line equivalents include `--output`, `--result`, `--after-export`,
`--output-mode`, `--fps`, `--units`, `--take`, and `--overwrite`.
Repeat `--collider`, `--soft-prop`, or `--collider-sequence` for multiple inputs.

## Limits

The solver uses a tetrahedral box cage over each soft prop's bounding volume,
then embeds the original surface in that cage. It suits solid cushions and
mattresses. Hollow objects, thin shells, disconnected pieces within one prop,
and heavily curved volumes are approximations. Each prop needs at least 1 mm
of extent on every axis. Multiple soft props have independent cages and do not
collide with one another. There is no soft-prop self-collision, gravity, wind,
water or grab tool. Closed, consistently oriented collision meshes work best.
Avoid deep initial intersections and very large jumps between animation frames.

## Window layout and shortcuts

Every Spin Tools window shares one layout and set of shortcuts; see the
[suite guide](../README.md#shared-window-layout). F1 or ? lists the keys.

## License

See [LICENSE](../LICENSE), [Additional Terms](../ADDITIONAL_TERMS.md) and [Notice](../NOTICE.md).
