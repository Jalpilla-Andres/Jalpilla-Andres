# Animated particle banner

The profile banner uses 2,300 travelling particles in the default configuration and is generated as pure SVG + SMIL. It is designed to make the morphing obvious: the portrait expands into a burst, reconstructs into a technical scene, then bursts again into the next scene.

## Main settings

Edit the top of `scripts/banner/generate.py`:

```python
SCENE_SEQUENCE = ("portrait", "ai", "network", "telemetry", "cyber")
TRANSITION_STYLE = "cinematic"
PORTRAIT_Y_BIAS = -34.0
```

The portrait remains the first and last scene. `PORTRAIT_Y_BIAS` controls the optical vertical position. Negative values move the portrait upward; positive values move it downward.

## Professional scene options

| Scene | What appears |
|---|---|
| `ai` | Hexagonal AI core, neural branches and orbital rings |
| `network` | 3D-style network globe with connected nodes |
| `telemetry` | Radar, signal sweep and telemetry waveform |
| `cyber` | Security shield, circuit traces and lock core |
| `terminal` | Command-line prompt and execution lines |
| `code` | Developer `</>` symbol |
| `java` | Java/JVM inspired glyph scene |

Examples:

```python
SCENE_SEQUENCE = ("portrait", "ai", "network", "cyber")
```

```python
SCENE_SEQUENCE = ("portrait", "telemetry", "terminal", "cyber")
```

```python
SCENE_SEQUENCE = ("portrait", "network", "ai", "telemetry", "cyber")
```

## Transition styles

### `cinematic`
Polished acceleration/deceleration with a strong particle burst. Recommended for a premium profile look.

### `energetic`
Larger particle explosion, faster snap and stronger rotational movement.

### `smooth`
More restrained movement while keeping the morph visible.

### `minimal`
Mostly direct morphing with a small burst.

## Generate locally

From the repository root:

```bash
python3 scripts/generate.py
```

This regenerates:

- `assets/banner-dark.svg`
- `assets/banner-light.svg`
- the scene `.npy` data under `scripts/banner/data/`

GitHub Actions regenerates the same files whenever the configured source/generator files change.
