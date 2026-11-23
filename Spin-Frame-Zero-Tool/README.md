# Current Frame to Zero

Author: **Spin256** · [spin256.com/tools](https://www.spin256.com/tools)

An independent DAZ Studio script for keeping the whole scene's current keyed
state at frame 0 and removing its other animation keys. It works without the
Drape/Soft Prop apps or Python.

1. Move DAZ to the frame you want to keep.
2. Run **Scripts > Spin Tools > Current Frame to Zero**.
3. The current state becomes frame 0, other keys are removed and the timeline is
   reduced to one frame. **Undo** restores the original animation in one step.

Install the menu entry by dragging **Install Frame Zero Menu.dsa** into DAZ's
viewport, or rerun the suite's **Install Spin Tools Menu.dsa** to register all
tools. You can also run **Current Frame to Zero.dsa** directly. No selection is
required: this operates on the entire scene, including hidden nodes, bone/pose
controls, keyed modifiers, materials, cameras and lights reached through their
element trees. It does not delete scene objects or save over your scene file.

**Fit To** and other node-reference properties are handled as references, rather
than being serialized as ordinary settings. Non-animated properties are left
alone. Animated numeric/color/string/image settings preserve their raw values,
so ERC and pose-control contributions are not applied twice. Unsupported keyed
properties abort before editing instead of silently losing animation.

This collapses ordinary property-keyframe animation. It does not bake external
simulation caches, aniBlocks, scripts or other procedural time-dependent systems
into static geometry. Those systems need their own baking first.

## License

See [LICENSE](../LICENSE), [Additional Terms](../ADDITIONAL_TERMS.md) and [Notice](../NOTICE.md).
