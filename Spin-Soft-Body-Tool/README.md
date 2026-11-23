# Spin Soft Body Tool

Author: **Spin256** · [spin256.com](https://spin256.com)

Subtle contact deformation for a posed body against props, another body, or
another region of itself. The user creates the overlap; the tool resolves it in
painted receiving regions while following a static pose or source animation. This is an artistic surface
solver, not a muscle/bone or full volumetric tissue simulation.

## Motion examples: read this first

Soft Body may catch some people out. A painted area normally follows its DAZ animation, with
contact and jiggle added on top. Once the area has a motion example, the simulation replaces the
DAZ animation inside it, so the example poses never appear in the final animation.

The first frame of the captured timeline is the starting reference. On other frames of the
animation, pose the area at its extremes: for breasts, up, down, left and right. Paint just that
area, for example the breasts, not neighboring parts that only move with the body's pose, such as
the shoulders. Save it as a named region, choose **Bounce** under **DAZ motion examples**, enter
the example frame numbers and click **Apply**.

The movement stays close to the range the examples show; contact can still push it a little
further. The edge fades into the unpainted skin over **Bounce edge blend** rings; raise it if the
edge creases. **Stiffness**, **Damping** and **Movement inertia** in the **Jiggle** section set
how the area swings, even with jiggle turned off.

## Open

Setup is described in the [suite guide](../README.md). An NVIDIA CUDA GPU is required.
Run **Open Spin Tools.cmd** at the suite root (or **Open Spin Soft Body Tool.cmd** here) for standalone OBJ/FBX
input, static OBJ/FBX export and baked FBX animation export. All files must already have the intended world
positions. OBJ is Y-up with selectable units; FBX axes/units are converted.

For DAZ, run the suite's **Install Spin Tools Menu.dsa** once, then use **Scripts > Spin
Tools > Spin Soft Body Tool**. Assign rigid collision objects if needed, then the soft bodies.
Collision objects are optional for self-contact or two receiving bodies. Object
assignments include only their own meshes, not automatically every child.
For a fitted piece that DAZ welds into its figure, leave **Keep fitted grafts
joined to their figure** enabled in the review step. Selecting the piece as a
soft body captures its assembled figure. Paint just the area to deform in the
helper. The return is one baked mesh with the join intact.

Choose a static current pose or enable timeline animation in the DAZ selector.
Timeline mode samples both soft bodies and moving collision objects at every
frame, then restores DAZ's original time. Standalone animation accepts animated
FBX bodies/colliders and collision OBJ sequence folders. Sampled animations must
have matching frame counts and frame rates; static objects can accompany them.
DAZ finalizes each animated node at its configured viewport quality before capture
and checks for consistent vertex/facet counts. Keep Resolution Level and SubDivision
Level constant across the source animation. Reset reuses the
current capture; start again from DAZ to record new frames.

After capture, DAZ opens the Soft Body receive prompt. Leave it open and click
**Send to DAZ** in the helper. The receive window closes after import. To use
DAZ before a result is ready, cancel the receive window. Opening **Spin Soft
Body Tool** from **Scripts > Spin Tools** always starts a new capture selection.
If a result is pending, the same dialog shows **Receive pending result** or
**Retry receive** with the error details. Choose it to import the saved result.
The helper stays open, so you can reset and run again
with the same paint and settings. Each successful revision hides the previous
result for that capture. Earlier copies remain in the scene.

## Clothing

Including clothing is not recommended. Soft Body has no cloth simulation: clothing assigned
as a soft body collides with the figure as another body, and clothing assigned as a
collision object pushes the figure's skin. Either way the skin and the clothing work
against each other.

Instead, animate the body here without its clothing and send the result to DAZ. Then, in
the [Drape Tool](../Spin-Drape-Tool/README.md), assign the returned body as a moving
collision object and the clothing as the garment, over the same frames. Leave the original
figure unassigned. For dense garments, turn on **Proxy simulation**.

A looped body is already moving on its first frame, so the clothing may start partly inside
it. Drape's initial settle often corrects this before the animation starts. If it does not,
run a static drape on the first frame, send the result to DAZ, and use that as the garment
for the animation. Only the garment's shape on the start frame is used.

## Paint the receiving area

The body starts protected. Nothing is deformed until you assign a receiving area.

1. Enable **Paint selection** and drag over the belly, thighs or another
   receiving region. Orange dots show the selection. **Remove from selection**
   erases it. Grow/Shrink adds or removes one topological ring per click.
2. Type a **Region name** and click **Save selection as region**. The controls under
   **Soft tissue** and **DAZ motion examples** apply to the chosen named region and are
   grayed out while Region is **-none-**.
3. Choose a tissue template and click **Apply template to region**. Blue marks
   the receiving region; gray remains protected. The template also fills in the
   run-wide **Deformation** values. Adjust **Paint softness** before applying to give a
   region a different softness. Applying with template **-none-** clears its softness.
   Clear the selection before painting another area; this does not remove
   assigned softness.
4. Painting starts off so left-drag orbits immediately. Uncheck painting again
   when you want to orbit after making selections. Reset view is always
   available. Window size and sidebar width are remembered.

To reselect a soft area, enable **Select soft patch (click)** and click it in
the preview. Shift-click adds another connected patch.

The brush follows connected mesh edges within its radius, preventing it from
jumping from a hand to an overlapped belly. It may stop at disconnected seams;
paint the other island separately. **Protect region** removes its softness without
creating a ghost. **Protect everything else** limits receiving areas to the region.
Clear all regions removes both softness and ghost assignments.

To let an external collider through part of the mesh, save that area as a region and use
**Ignore collider contact on region**. Purple marks the pass-through area.
Paint softness can overlap this mask, so nearby deformation and jiggle can
still move the area. Use **Restore collider contact on region** to undo the
mask locally. **Clear all regions** also clears the pass-through mask.

For an animated soft seal, paint the backing as pass-through, leave the lips
soft with collider contact enabled, and turn on **Deformation → Part soft seal
around moving collider**. When the moving collider penetrates the soft edge of
the mask, it can push that edge sideways across its path. Normal contact still
shapes the surrounding surface. The mask leaves the original mesh intact.

For a piece cut out of a larger mesh, enable **Pin open mesh edges** in
**Deformation**. Every open border, including holes, follows the source
pose while the painted interior deforms. Increase **Pinned border rings** to
hold adjacent rows if the join needs a wider fixed strip. The **Protected boundary blend**
setting controls how gradually the interior starts moving near the pinned edge.
For a closed piece with a capped cut, save the seam as a region and use **Protect region**.
If an actual hole needs to widen, leave its rim soft and use **Protect region**
on the outer join instead of pinning every open edge. **Opening stretch** allows
more movement near soft open edges and the rim of a painted pass-through area.
The setting tapers into the surrounding mesh. The deformation limit and face
protection still apply.

### Suggested areas and named regions

In an animated session, type example frame numbers in **Region painting →
Example frames** and click **Suggest moving areas**. Areas whose shape changes
on those frames, compared with the first frame, show in orange; moving or
turning a limb as a whole is ignored. Enable **Pick suggested area (click)** and
click one to select it (Shift adds another).

Type a **Region name** and click **Save selection as region**. Under **Named
regions**, choosing a region selects exactly its vertices and recalls the
softness, collider contact and motion example applied to it. Choose **-none-**
to clear the selection. Painting, growing or shrinking leaves the region, so the
box returns to -none-. Regions are kept while the window is open.

### Collider contact

When a collider is part of the session, **Collider contact** lets you paint
areas of the collider that should not push the soft body (shown in red), for
example the solid part behind the lips when the bodies are swapped for a second
pass. Enable **Paint collider no-contact (left drag)** and brush over the
collider; **Restore contact instead** erases and **Clear collider paint** removes
it all. The brush radius is the one in Region painting. A face stops pushing only
where all three corners are painted, so paint slightly past the edge.

## Hand pressing into the same figure

Paint the receiving belly, save it as a region and apply softness. Then clear the selection,
paint the hand, grow as necessary, save it as a region and choose **Make region rigid / ghost**. The original
hand remains rigid and acts as the internal collider. In animation it follows its
prescribed posed movement rather than remaining fixed in world space.
Neighboring faces are excluded from self-contact to avoid the hand pushing its
own wrist. Protected-boundary blending tapers mobility near the receiving boundary.
You do not need to duplicate the figure in DAZ.

For crossed legs, paint both receiving regions and apply softness to both. Their
relative softness controls how they share the correction. Paint knees or other
firm areas with a lower softness, or protect them. Soft pairs use their current
surfaces during the solve, rather than each resolving the whole overlap against
an unchanged copy of the other side.

## Settings and preview

- **Paint softness** is stored on the region when **Apply template to region** is
  clicked. Firm regions yield less relative to softer regions in a shared contact.
- **Maximum indentation** limits displacement from the captured pose. This also
  limits the self-contact search distance; avoid overlap deeper than this setting.
  Firm templates use a smaller limit. The limit is global for the run.
- **Spread / smoothing** carries deformation into neighboring selected vertices.
- **Shape recovery** pulls back toward the captured shape between contact solves.
- **Contact clearance** is the requested separation in millimeters.
- **Protected boundary blend** tapers mobility on the receiving side of a protected
  boundary over the chosen number of topological rings.
- **Steps per run / resume** controls each solve batch. It is an iteration count,
  not physical elapsed time. The worker pauses at the end and Resume continues
  the existing deformation. This setting is used only in static mode. Reset
  restores the starting pose and retains masks.

The control sections collapse by clicking their title, and remember their open
state. Region painting starts open; advanced sections start
closed. Painting remains off whenever a new window opens.

## Animation and jiggle

Paint receiving regions on the first pose, just as for a static solve. Protected
vertices and rigid ghost regions follow the source animation exactly. Soft
deformation is added relative to each sampled pose; the original pose itself is
not simulated. Both body and collider meshes must keep their vertex order and
topology throughout the sequence.

After a run finishes, click **Replay** or drag the frame slider to inspect the
recorded body and collider motion. Replay uses the source frame rate. Sending
to DAZ or exporting still uses the entire recorded sequence.

### DAZ motion examples

Capture a timeline containing the original poses and the authored examples.
The first frame of the timeline is the rest (reference) shape. Select a soft
named region, choose a behavior under **DAZ motion examples**, type its example
frames (leave the box empty to use the Region painting frames) and click
**Apply**. Choosing the region again recalls the example; Apply with behavior **-none-**
removes it. Regions with examples must not overlap.

- **Bounce** uses frames that show the painted part pushed different ways. The
  solver blends between them as the body moves, so the tissue lags and settles
  within about the range of the examples. **Bounce edge blend** sets how many
  rings of the painted area fade into the unpainted skin (8); raise it if the edge
  creases, lower it for a small area. Stiffness, Damping and Movement inertia in
  **Jiggle** set the swing, even with jiggle off.
- **Opening** uses frames with the collider in different places (in, out, left,
  right). The solver blends them by where the collider is and which way it
  moves. With a single wide frame, contact drives how far it opens.
  **Fold rounding** softens compressed creases (25% by default). **Collider
  drag** sets how far the lips follow the collider along its surface (3 mm).
  **Drag reach** sets how far that tug fades along the painted mesh (8 mm).
  Set either slider to zero to disable it.

Jiggle and Opening also work without examples; examples make the motion follow
the shapes you authored. If the example frames also reshape skin joined to the
region, Apply warns and suggests how many times to grow the region, because that
skin would otherwise jump on those frames.

### Finish on last frame, or loop

Under **Animation loop**, choose **None** or one of:

- **Finish on last frame**: over the last **Blend frames** (default 8) the painted areas ease onto their DAZ
  shape on the last frame and come to rest there, so the animation ends exactly as you keyed it (as it starts from
  a still pose). It does not need the body to end where it starts; replay plays straight through.
- **Loop animation (carry motion)**: for a repeating animation. After the last frame the simulation takes one
  real step from the last pose into frame 0 and carries on into frames 1, 2, ... on the same body poses until it
  matches the first run (within 0.2 mm), which is used from there; frame 0 is that simulated step, so the tissue
  never stops at the join. If they have not met by the middle of the animation, the closest frame is used and
  the second pass is steered onto it over the Blend frames. Make the avatar's last pose the one *before* its first
  (a repeated first frame just holds for one frame). Replay repeats it, playing every frame.

### Manual frame guides

To steer the result, pause on a frame and use **Paint selection** to select
vertices already assigned softness. In **Frame guides**, enable **Left-drag
selection to set guide** and drag in the viewport. **Guide pull radius** spreads
the pull along the surrounding soft surface, with less movement farther from
the selection. A radius of zero pulls only the selection. Release to save the
guide. You can clear the selection and still see the saved edit at that frame.
Replay shows the last simulation. Add guides on more frames, then choose
**Run with guides**. Targets ease between keys for the same vertices.
**Guide approach** starts the motion before the first key (18 frames by default).
**Guide release** eases out after the last key (12 frames by default).
**Guide strength** sets how much of the target displacement the guide applies.
Jiggle continues around the moving guide. Collision and protected vertices still
apply. Guides stay with the
current helper session, including after Reset.

Use the original animated figures as inputs, not static mesh copies returned by
an earlier solve. Those copies have no rig and do not inherit later pose changes.

Contact deformation/recovery works without jiggle: a moving hand can press in
and release, and **Shape recovery** controls how quickly the dent fades. Zero
recovery retains the offset until contact or other constraints change it.

Open **Jiggle** and tick **Enable motion-driven jiggle** for inertia,
overshoot and settling in the painted regions.

**Jiggle preset** provides settings for quick motion, slower
motion, and lag through the turnaround. It enables jiggle and volume
preservation. Gravity and painted softness are set independently.

- **Stiffness / bounce frequency** controls spring return speed in Hz.
- **Damping** removes oscillation. Increase it for less bounce.
- **Movement inertia** controls lag caused by changes in prescribed motion.
  Softness painting also scales the inertial response. Zero inertia disables
  motion-driven lag; contact can still excite the spring.
- **Jiggle gravity** adds downward loading to the painted jiggle region.
  It defaults to 0%. At 100%, the acceleration is Earth's gravity. Include
  a stationary lead-in if the tissue should already be sagging before motion.

With jiggle enabled, spring stiffness/damping replace the simple contact-recovery
decay. Deformation spread and surface protection continue to apply. These are
artist controls, not calibrated tissue parameters or a skeleton/force simulator.

Each source frame advances by exactly its elapsed time. Dynamics are substepped
to at least 120 Hz, with moving geometry interpolated between sampled poses.
No extra settling seconds or extra exported frames are inserted between poses.
Include a hold at the end of your source timeline if you want time for jiggle to
settle before the sequence ends. **Animation collision accuracy** in Deformation
increases contact projection work per substep; higher values cost more processing
and can help contact converge, without changing the animation duration.

Pause/resume retains state. Export becomes available only after the complete
sequence has finished; paused partial sequences cannot be returned. Output is
baked geometry animation with the source start/end frames and sample rate,
not a rigged character or editable skeletal performance.

The **View** categories are Soft bodies, Props / collision objects, and Ghost
colliders. Each has independent visibility and opacity, remembered between runs.
Ghost faces are removed from the original body's preview and drawn only as ghosts,
so two transparent copies do not compete. A complete triangle becomes a ghost
when all three of its vertices are marked rigid. Visibility never changes collision
participation or export. Orange selection markers remain separate editing guides.

Space runs/pauses/resumes; V resets the view; Ctrl+Enter exports an available result.
Region/settings inputs are disabled while the worker is active, including paused.
After a run they can be edited again; the next Run starts from the first frame.
View controls remain available.

Surface protection is always enabled: contact steps scale to local edge size,
neighboring displacement changes are limited, and triangle constraints resist
folding/collapse. A final safety check reduces an unsafe step before previewing
or exporting it. If no safe progress is possible, static mode pauses on its last
valid shape; animation retains a safe deformation for that pose and continues
the source timeline. A surface-protection message identifies limited corrections.
Paint a wider receiving region or reduce
the initial overlap, then reset and rerun. Complete overlap removal is secondary
to keeping a usable surface; deep/narrow contact may remain unresolved.

## Return and limits

For static output, pause or wait for the batch to pause, then **Send static mesh
to DAZ** or export. For animation, wait for completion and **Send baked animation
to DAZ** or export FBX. Static output is the displayed snapshot; animation contains
every completed source frame. Both retain original vertex order and
OBJ topology/UV/material records. Ghost faces are retained in output at their
prescribed positions. The preview hiding is not a geometry deletion.

**DAZ return creates static mesh copies or baked mesh animation and retains the
original figures hidden. It does not create a morph or preserve a rig on the returned copy.**
The original rigged figures remain available in DAZ.

**DAZ return format** in the Animation loop section chooses **DAZ native (DUF)** (the default,
fast to load) or **FBX (compatibility)** for an animation sent to DAZ.

No Genesis-specific softness maps, automatic body-part detection, pressure or
body-weight loading are implemented. The supplied poses/movement create contact.
This is sampled vertex/triangle collision with a local facing-normal test, not
continuous collision detection. It works best with consistently oriented meshes,
small overlaps and sufficiently detailed surfaces. Protected regions or the
indentation limit can prevent complete overlap removal. Outward tissue bulge is not
guaranteed. Use preview to assess the result.

## Standalone and bridges

The launcher accepts `--soft-body FILE` and optional repeated `--collider FILE`,
`--units cm|m|mm`, or `--project FILE.json`. A version-1 project JSON uses
`soft_bodies` and `colliders` arrays, `output_mode` (`static` or `animation`), `fps`,
optional `collider_sequences` folders, `take`, `output`,
`result`, `after_export` or a callback argument array containing `{result}`.
JSON paths are relative to the JSON file. The helper still opens for painting.
Callbacks run only after a complete export manifest has been written.

## Window layout and shortcuts

Every Spin Tools window shares one layout and set of shortcuts; see the
[suite guide](../README.md#shared-window-layout). F1 or ? lists the keys.

## License

See [LICENSE](../LICENSE), [Additional Terms](../ADDITIONAL_TERMS.md) and [Notice](../NOTICE.md).
