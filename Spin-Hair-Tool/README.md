# Spin Hair Tool

Author: **Spin256** · [spin256.com/tools](https://www.spin256.com/tools)

Settles card hair against the avatar on an NVIDIA GPU while keeping the
hairstyle, for a still pose or a timeline animation, from DAZ Studio or from OBJ/FBX
files (see **Standalone**). Part of the Spin Tools suite; it uses the suite's shared
Python environment and receive window.

## Use

1. In DAZ, open **Scripts > Spin Tools > Hair Tool...**.
2. Assign independent hats, pillows and other props as **Collision objects**, choose exactly
   one character as **Avatar**, and assign the hair as **Hair**.
   Assign the main figure first; additional selections in the Avatar stage become
   **Avatar attached meshes**, for fitted clothing and accessories. They follow the
   same body surface where correspondence stays close at both endpoint poses.
   Independent Collision objects retain their own paths.
   If the product has separate scalp and hair nodes, select and assign both as **Hair**.
   Hair is optional: an avatar on its own can be captured to grow hair on it (see **Grow hair**).
3. In the helper, freeze anything that should stay as styled, then **Run hair on GPU**.
   It settles for the **Initial settle** time and pauses. **Resume** (or **Space**)
   then keeps simulating until you press **Pause** (or **Space**) — use this for
   wind, water and pulling.
4. **Send to DAZ**. The Spin Tools receive window imports "<hair> - Simulated" and
   hides the original. The helper minimizes so you can adjust and send again;
   tick **Close after sending** (remembered) to exit instead.

## How it works

Line-based hair without polygon geometry (such as MRL DFLL Hair) is labeled
**Unsupported** in the export dialog and cannot be assigned. Polygon-card hair
remains supported. Separate selected scalp and polygon-hair objects are simulated
together and returned separately with their own materials. The optional extra
scalp-cap generator is available for a single hair mesh; a separately selected
scalp is included directly.

Each hair card (a ribbon of quads) becomes one strand: a chain of points along
its center line. The GPU settles the chains (stretch, bending, style hold,
collisions) and every card vertex is rebuilt along its strand, so the returned
mesh keeps its exact vertex order, UVs and surfaces.

- **Roots**: a strand whose end lies on the scalp (a surface named scalp/cap, or
  the avatar's head) is fixed there. A strand starting on other hair (layered or
  tied hair) hangs from the closest strand, preferring scalp-rooted ones.
  When both ends lie near the scalp, the tool can learn root-to-tip texture
  direction from confidently rooted cards on the same surface. It needs at least
  eight examples with at least 90% agreement; missing or conflicting texture
  coordinates leave the geometric inference unchanged. This uses no hairstyle
  names or world-height rule, so swept-up hair can keep its lower root. Nearby
  supporting strands are preferred at their scalp-side points, not returning tips.
- **Avatar and Collision objects** describe purpose, not movement. The avatar
  carries the scalp roots and uses solid-body collision. Collision objects are
  double-sided barriers. Connecting strand segments and broad ribbons can still
  pass through a little. Both roles are
  captured throughout a timeline, so a hat can move independently of the avatar.
- **Stiffness** (default 90%) is the hair's own stiffness over its whole length:
  each strand keeps its shape in short and long sections that turn freely, so hair
  holds an arc against gravity without being pinned anywhere. 0% hangs like string.
- **Trapped contact distance** (default 10 mm; 0 disables): hair within this
  distance of two opposing collider surfaces keeps a persistent contact, rather
  than repeatedly being pushed between them. The contact follows the surfaces
  and releases when their gap opens. This applies to any collider pair, without
  identifying hats or scalp. Reset before changing this setting. It does not
  guarantee containment of every wide ribbon reconstructed around a strand.
- **Style at rest** (default 100%): how much of the DAZ style an upright figure keeps
  under normal gravity with nothing else acting. These strands stand for a full head
  of hair that holds them up, so that share of their weight is carried, along gravity
  and only up to the 1 g the style was made under. "Down" for the style is taken from
  the head (fitted to the scalp roots), so on a figure leaning or bending over the
  support fades and the hair drops toward the floor. Leaning, stronger gravity, wind and pulling still move the hair; nothing is
  pinned in place. 0% hangs freely.
- **Weightless lift** (default 60%): without weight, hair relaxes outward from the
  head, leaving the scalp toward 90 degrees while keeping its curves (each strand
  leans a little differently). It grows as hair loses weight: none at normal gravity
  or more, full at zero gravity. Under water, buoyancy carries the hair's weight point
  by point, so submerged lengths bloom while hair above the waterline keeps its
  style. Tied hair blooms about its tie. A wind setting doubles as a water current:
  under water, drag acts against the moving water, so a current carries the hair.
- **Water from the scene**, as in the Drape Tool: a captured collision object named
  "Water..." that is a flat horizontal plane is the water surface, not a solid. It
  switches water on at its height. Without one, turn water on in the Water panel and
  tick "Return water-height plane to DAZ" to get a plain reference plane back with the
  hair.
- **Scalp cap (for extreme poses)** (off at every start): hair assets are made to be
  seen upright and often leave part of the scalp without strands. With the cap on,
  the tool lays short cards (about 7 cm) combed down over the nominal hair area:
  the crown, back and sides of the head above the ears (not the ears, face or
  neck), whether or not real strands start there, each a copy of the root end of the nearest real card, so it keeps that card's UVs
  and material. Cap cards are fixed to the scalp and not simulated (they stay flat
  when a figure bends over) and are added after the original polygons in the result.
  **Cap density** (default 100%, about 1,000 cards on a long-hair asset) adds more
  if skin still shows through the see-through hair shader. Toggling either rebuilds
  the strands (about 20 s); freezes are kept. Leave it off for ordinary poses.

The window remembers its position, size and panel width, like the other Spin tools.
- **Hair-to-hair collision** (on by default, **Hair spacing** 0.6 cm) keeps strands
  from sinking into each other; a pair never has to be further apart than it was as
  captured, so nothing puffs up.
- Strands starting at frozen hair (a tie or bun) hang from it. Near each root the
  styled angle and shape are held a little (fixed defaults, not in the window), so
  hair leaves the scalp as styled.
- Selecting: a plain click replaces the selection, **Ctrl**-click adds to it and **Alt**-click removes from it
  (Pick strands and Ring select).
- **Pin both ends** holds the selected strands at the scalp and at their far end (hair swept back into a tie or
  tucked in): the hair in between still moves. Pinned strands show cyan; **Unpin selected** releases them.
- **Freeze** whole strands that should not move, such as hair wrapped around a
  ponytail tie. Strand ends touching frozen hair are held by it, so the tie keeps
  the ponytail and swept-up hair in place. Use **X-ray** and **Ring select** to
  reach ties hidden inside the hair, or **Select surface** for a whole surface.
- Strand colors: blue scalp-rooted, green hanging from other hair, red free
  (nothing holds them; freeze them or accept that they fall), purple frozen,
  yellow selected.

Pieces that are not hair cards (the scalp cap, beads, other meshes) stay as captured.
Tapered ribbons and ribbon layers touching at a vertex are recognized as strands
when their shape supports a center line. The setup note reports any pieces that
still remain static.

## Grow hair

**Grow hair** adds strands to areas you paint on the avatar: short hair such as a beard,
stubble, armpits or a filled-in scalp, or longer hair. It works on still captures (without
**Use timeline movement**). To grow hair on an avatar that has none, assign only the avatar
in DAZ and answer Yes when asked to capture it on its own.

1. Type a name for the area, such as Beard, and click **New area**.
2. **Paint polygons**: left-drag on the avatar to add, **Alt**-drag to remove. **Brush
   radius** sets the brush; **Clear area's polygons** starts again. Painted polygons show
   in pink. Painting turns on by itself for a new area.
3. Choose the area's settings and material.
4. **Grow all areas** grows every painted area, replacing hair grown before. After changing
   an area, click it again (press **Reset** first if a run is active).
5. **Run hair on GPU** and **Send to DAZ** as usual. Each area returns as its own prop,
   named after the area, with plain-colored surfaces you can restyle in DAZ.

Strands come straight out of the skin, root on it and are simulated with the rest of the
hair. Growing again with the same settings gives the same strands. The scalp cap turns off
when hair is grown.

| Setting | What it does |
|---|---|
| **Edge** | **Smooth line** (default) rounds off the painted outline. **Polygon edge** keeps roots on the painted polygons. **Circle**, **Ellipse**, **Rectangle** and **Triangle** fit a shape to the area; these suit compact areas facing one way. |
| **Point the other way** | Triangle only: turns the triangle around. |
| **Allow past the painting** | Shapes only: lets the shape reach nearby skin facing the same way. Off, it only trims the painted polygons. |
| **Feather** | How far the blend reaches from the edge toward the center. 0% is a hard edge. |
| **Edge density**, **Edge length** | Density and length at the very edge, as a share of the area's own. |
| **Scatter** | Even (spaced), Random or Regular. |
| **Density** | Strands per square centimeter. |
| **Length**, **Length variation**, **Minimum length** | Strand length, how much shorter strands may be at random, and the shortest allowed. |
| **Segments** | Bends along each strand. Long hair needs about one per 2.5 cm to follow the head; the panel warns and suggests a number. |
| **Width** | Width at the root; strands narrow to the tip. |
| **Stiffness** | Share of the simulation's Stiffness these strands keep. At 0% (default) they fall and lie under full gravity; higher stays springy. |
| **Root lift** | How firmly strands keep leaving the skin straight out before falling. Gives volume and covers a parting; 0% lies flat from the root. |
| **Material**, **Secondary**, **Secondary mix** | Named materials and the share of strands that take the secondary one, such as gray hints in dark hair. **Edit…** renames or recolors a material in every area that uses it; **New material…** adds one. |

Hair longer than 60 cm or with more than 30 segments makes many polygons and can be slow to
simulate and return; the tool asks before growing it.

**Grow presets** keep every area, its painting, settings and materials under a name (saved
in `grow_presets.json` in this tool's folder). Areas belong to the capture they were painted
on, so a new export from DAZ starts with none: choose a preset to load its areas, or
**Apply preset** to load and grow them in one step. A preset works on figures with the same
mesh as the one it was painted on.

## Standalone

Double-click **Open Spin Tools.cmd** at the suite root and choose **Spin Hair Tool**, or
**Open Spin Hair Tool.cmd** in this folder. Choose the hair (OBJ or FBX),
the avatar it grows from and any other collision objects, the OBJ units and **Still** or
**Animation**. For animation the avatar is an animated FBX or a folder of numbered OBJ
frames (the same vertex order in every frame). OBJ uses Y-up world coordinates; FBX axes
and units are converted. The window then works as it does from DAZ; **Export…** writes the
hair as OBJ or FBX (animation: FBX in centimeters) with `name.result.json` beside it.

What the hair needs:

- **Card hair**: strips of quads (ribbons), as DAZ hair uses. Curve or strand hair with no
  polygons is not supported.
- **Roots**: a surface (OBJ material) named *scalp*, *cap* or *skull* is the scalp: it is held
  still and the hair roots on it. A wide shell lying on the head with another name is also
  recognized as the scalp. Without either, each hair object roots on the avatar's skin at
  the strand end nearest it. If only some of a hair object's strands reach a scalp, the rest
  hang from nearby hair, so split a combined hair object into one file per hairstyle.
- **Units and orientation**: check that the hair sits on the avatar's head in the first view;
  a wrong unit or up axis puts it far off or tiny.
- **Materials**: the OBJ's material file is kept where it is; textures are not copied.

Command line (from this folder):

```text
"Open Spin Hair Tool.cmd" --hair hair.obj --avatar avatar.fbx --output-mode animation --fps 30
```

Other options: `--avatar-sequence FOLDER`, `--collider FILE` (repeatable), `--units cm|m|mm`,
`--take NAME`, `--output FILE`, `--result FILE`, `--overwrite`, `--close-after on|off`,
`--after-export SCRIPT.py` (runs `python SCRIPT.py RESULT.json` after each export, no shell),
and `--project FILE.json` with the same keys (`hair`, `avatar`, `avatar_sequence`, `colliders`,
`units`, `fps`, `output_mode`, `take`, `output`, `result`, `after_export` or a `callback`
argument array containing `{result}`); its paths are relative to the JSON file.

With a hair and an avatar on the command line the scene opens straight away. With `--output`
the export writes there without asking. With a callback the button reads **Send…** and
**Close after send** is on by default; otherwise **Close after export** is off. Each export
writes the files first, then `name.result.json` (status, units, frame range and each output
file), and only then starts the callback, so a receiver can read the result safely.

## Gravity, wind, water and live pulling

- **Gravity**: aim it with the sphere or angles and set its strength (0 g to 5 g),
  as in the Drape Tool.
- **Wind**: the same source, beam, spread, gusts, turbulence and avatar shelter as
  the Drape Tool. It starts in front of the hair, blowing at it; after turning it,
  use **Aim at hair center**. Hair catches the wind across each strand.
  Aiming uses a stable scalp/root pivot, independent of loose hair or live pulls.
  For a static timeline result, the default source and target are placed in the
  final head pose automatically when the hair is prepared, without resetting the
  source settings. Animation
  output has **Track avatar during animation** (remembered, off by default):
  source position and direction keep their offset relative to the avatar's head.
  The source fields and overlay show that moving world-space position. With
  tracking off, wind stays fixed in the scene.
- **Water**: set the waterline; hair below it is buoyed (100% is neutral, more
  floats) and slowed, with extra resistance entering the surface.
- **Live pulling**: switch on **Pull hair** and drag while running or paused.
  Paused pulls edit the current result without advancing the timeline, applying
  wind/gravity or restoring the captured style. Strand lengths, a gentle curve
  constraint and collision checks keep the edit connected. Frozen hair
  and scalp roots remain fixed. Wide ribbons can still pass through a hat.
  **Adjust (free on resume)** leaves the edit in place until physics resumes;
  **Hold after pulling** retains the pulled points during simulation. Holds follow
  the head. **Release holds** frees them without changing the paused result;
  Reset removes all holds. Sending the paused result includes the completed edit.
  Pulling remains disabled when recording an animation return.
- **Wind catch** (Wind section, default 100%) sets how strongly wind grips the
  hair. Higher streams it out flat; lower lets gravity keep more shape, so a
  ponytail arcs from the tie. **Friction** is 0 by default: hair slides freely off
  the shoulders and back.
- **Turbulence** (Wind panel, default 50%) is flutter: small eddies carried
  downwind, out of phase from strand to strand, so each strand whips on its own,
  including in the wake behind the head. High wind (around 15 m/s) streams the
  whole mass back with fluttering ends.
- **Wind shelter** (Wind section) sets how much the avatar blocks wind to hair
  behind it, with shelter on in the Wind panel. Air flows around a head, so the
  default lets 40% through; a ponytail behind the head still catches the wind.

Pause to change gravity, wind or water, then resume from the current hair.

For timeline captures, **Avatar speed limit** works like the Drape Tool: fast
pose changes are split into smaller moves, giving the hair more simulation time
to follow. Default **0.75 cm**; lower is slower, **0** preserves DAZ timing.
It measures movement across the avatar and all animated collision objects and
uses one timing schedule for them. It works with two captured poses as well as
longer sequences. Between poses, body parts rotate in arcs using the Drape Tool's
rigid-limb interpolation; their lengths are preserved instead of shrinking through
straight vertex blends. The speed limit measures those arcs. Captured endpoint
poses remain exact; genuine nonrigid changes in the source are retained.
Local muscle/skin shape changes blend in the rotating body-part frame, rather
than blending their offsets in world space. Small shape changes within a limb
are tolerated when grouping patches, reducing artificial extra joints.
Change it after Reset; the value is remembered. Animation
returns retain the original DAZ frame numbers and frame rate, with the slowed
internal simulation sampled at each source pose. It does not lengthen the DAZ
animation itself.

View: **Hair opacity**, **Avatar opacity** (figure and attached meshes), and
**Collision objects opacity** are separate, remembered preview settings.
Independent collision objects also have their own show/hide control; hidden
objects still participate in simulation.

## Keyboard shortcuts

Shared with the other Spin tools: `keybindings.json` in the Spin Tools folder (`common`
keys plus a `hair` section). Edit it and reopen the window; hover the note in Select and
freeze for the loaded keys. Defaults: Space pause/resume; W A S D orbit, arrow keys pan,
PgUp/PgDown zoom, V reset view; P pick strands, R ring select, + / - grow / shrink the
selection, Ctrl+D clear it; F / Shift+F freeze selected / unselected, U / Shift+U unfreeze
selected / all; G pull hair; X X-ray; Esc leaves pick or pull mode; Ctrl+Return sends to
DAZ. Keys are ignored while typing in a number box.

## Animation (timeline movement)

As in the Drape Tool: tick **Use timeline movement** in the DAZ dialog and choose the start
and end frames (and, if you like, **Prefer an animation sequence on return**). DAZ captures
the hair at the start frame and the avatar at every frame.

In the Hair window, Run settles the hair at the start frame for the initial settle time,
then plays the movement. Frozen hair and pinned ends follow the spot of the avatar's scalp
they sit on. Scalp roots, frozen hair (ties, buns), the hair's scalp mesh and the
scalp cap ride with the head; everything else simulates against the moving avatar, with
gravity, wind, water and pulling as usual. **Return preference** chooses:

- **Final frame (still hair)**: the movement brings the hair to the end frame; you can keep
  settling, adjust and pull, then send the still hair for that frame.
- **Animation (every frame)**: each DAZ frame is recorded during playback and sent as an
  animated hair (see **Sending an animation to DAZ**). As in the
  Drape Tool, an animation is recorded in one uninterrupted run: live pulling is off and
  a paused or finished run cannot be resumed (Reset, then Run again). The scalp cap is not
  available for animations. Changing the return choice resets the run.

Hair carries its momentum: strands are stiff along their length but swing about their
roots, so when the figure leans or turns quickly the hair lags, lifts and swings past
before settling. **Momentum** (shown for timeline captures, default 50%) eases the
stiffness of hair moving fast relative to its root while the movement plays, so a flick of
the head throws the hair up and over instead of the stiffness pulling it back; it springs
back to its style as it slows. 0% keeps full stiffness throughout.

The avatar's frames are read in the background while the hair settles.

## Presets: still poses and animation

The **Preset** box at the top of **Simulation** sets how the hair behaves. **Static** gives free, lively hair for a
still pose you refine by hand. **Animation** (chosen when the return is Animation) gives hair that moves together and
stays near its style on every frame: **Clumping** (50%) makes the hair around each lock move as one, **Head drag**
(6 /s) lets hair whipping about relative to the head settle while hair moving with the head is not held back, and
Momentum is off. Changing any of these settings makes the preset **Custom**; **Save preset…** keeps your settings
under a name (saved in `hair_presets.json` in this tool's folder, available next time), and **Delete preset**
removes a saved one.

## Sending an animation to DAZ

**DAZ return format** in the Animation section chooses how an animated hair goes back to DAZ. **DAZ native (DUF)**
(the default) sends it in DAZ's own format and loads in seconds even for a large hair; **FBX (compatibility)**
imports far more slowly for large hair (about an hour for 580,000 vertices). Either
way the returned hair keeps the original's materials, is checked against the simulation and replaces the original
only after the checks pass.

## Seamless loop

**Loop animation (carry motion)** makes a timeline animation repeat seamlessly. A simulation starts from rest on
its first frame, so a plain loop restarts the hair's motion at the join. With this on, the hair keeps simulating
past the last frame with its own momentum into frames 1, 2, ... on the same avatar poses; frame 0 becomes the last
frame. The last frame leads into frame 0 with one real step, so make the avatar's last pose the one *before* its
first (don't repeat the first frame at the end; a repeat just holds for one frame). From frame 1 to the middle of the animation, where the first run is moving too, the hair is gently eased onto the first run, which is used from the middle on. It applies to the Animation return and costs about half the animation again.

**Warm-up loops** (default 2) plays the movement that many times before recording, carrying the hair from each
loop's end into the next start, so the recorded loop starts with the hair already swinging. The finished status
says how much the hair still changed between the last two warm-up loops; if that is large, add loops or calm the
hair (less Momentum, more Damping). Each warm-up loop costs one more run of the movement.

## Window layout and shortcuts

Every Spin Tools window shares one layout and set of shortcuts; see the
[suite guide](../README.md#shared-window-layout). F1 or ? lists the keys.

## License

See [LICENSE](../LICENSE), [Additional Terms](../ADDITIONAL_TERMS.md) and [Notice](../NOTICE.md).
