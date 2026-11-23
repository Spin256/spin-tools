# Spin Drape Tool

> Part of the **Spin Tools** suite: keep this folder inside it. Setup is described in the
> [suite guide](../README.md).

Author: **Spin256** · Website: [spin256.com/tools](https://www.spin256.com/tools)

Double-click **Open Spin Drape Tool.cmd** to import a scene, drape its garments and
export OBJ/FBX. Animated FBX and OBJ sequences can supply moving collision objects.

A garment can be opened on its own. Skip the collision-object stages when none
are needed; only the garment is required. Without scene colliders, cloth can use
the helper's ground plane or simulate freely with gravity, wind and water.

For DAZ Studio, run the suite's **Install Spin Tools Menu.dsa** once, then use
**Scripts > Spin Tools > Spin Drape Tool...**

An NVIDIA CUDA GPU is required. See the [standalone and custom bridge guide](Standalone%20guide.md)
for file formats, the command line, result JSON and after-export callbacks.

## Seamless loop

**Loop animation (carry motion)** makes a timeline animation repeat seamlessly. A
simulation starts from rest on its first frame, so a plain loop restarts the motion at the join. With this
on, the simulation keeps going past the last frame into frames 1, 2, ... on the same avatar poses until it
meets the first run (within 0.2 mm), and the first run is used from there; frame 0 is that simulated step.
If they have not met by the middle of the animation, the closest frame is used and the second pass is steered
onto it over **Loop landing frames** (default 8). The finished status says where they met. The avatar
animation's last pose should be the one *before* its first: after it the simulation takes one real step into frame 0 (a repeated first frame just holds for one frame). It costs up to half the animation again in simulation time. Final settling is skipped while looping.

## Heavy garments
For very dense garments, tick **Proxy simulation** in the Simulation section: a lighter copy (size
adjustable) is simulated and its motion carried onto the full garment, which is what is sent back to
DAZ. New wrinkles are only as fine as the proxy; captured detail is kept. The status bar tells you,
from how fast earlier runs were on your PC, when a garment will be slow and the proxy is worth it.

## Ground
Simulation > Ground adds a floor the fabric rests on: DAZ's floor, the avatar's lowest point (over the
whole timeline), a flat prop you captured whose name starts with "Ground", or your own height. Ground
grip sets whether fabric slides or stays where it lands. Tick "Send ground plane to DAZ" to get a
matching plane back with the result, like the water plane.

## Window layout and shortcuts

Every Spin Tools window shares one layout and set of shortcuts; see the
[suite guide](../README.md#shared-window-layout). F1 or ? lists the keys.

## License

See [LICENSE](../LICENSE), [Additional Terms](../ADDITIONAL_TERMS.md) and [Notice](../NOTICE.md).
