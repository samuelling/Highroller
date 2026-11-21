# Highroller

A crayoned-in coloring sheet, cut into its parts and brought to life in WebGL.

Open `index.html` from any static server (or GitHub Pages).

- **Trip** slider: all the way left is the untouched sheet; right adds motion, hue cycling, rainbow auras, lighthouse beams, and at the top end a feedback "melt".
- **Tap** the plane, gull, lobster, fizz, cooler, surfer, birds, or letters to set them off.
- **Parts** shows the segmentation as an exploded view. **Kaleido** folds the whole scene into a mandala. **Sound** adds surf, a drone, and plucks on taps.

## How it was made

`tools/` holds the pipeline that produced `assets/`:

1. `seg.py` — Segment Anything (ViT-L) with box + point prompts from `prompts.json`, one mask per object.
2. `export.py` — cleans the masks, pulls the ten title letters and the three birds out by colour, fills the holes left behind with LaMa inpainting, and writes the background, 20 sprite layers, and two region maps (water / sun / cloud, lighthouse / house).

Everything in `index.html` is a single file with no dependencies.
