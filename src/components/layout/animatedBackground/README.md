# DarkVeil

A WebGL-powered animated background component using neural network-generated patterns (CPPN) with customizable visual effects.

## Usage

### Importing

```tsx
import { DarkVeil } from './components/DarkVeil';
```

### Basic Example

```tsx
function App() {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <DarkVeil />
    </div>
  );
}
```

### With Custom Props

```tsx
<DarkVeil 
  hueShift={180}
  speed={1.5}
  warpAmount={0.3}
  resolutionScale={0.75}
/>
```

## Props

All props are optional.

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `hueShift` | `number` | `0` | Shifts the color hue in degrees (0-360). Use this to change the overall color scheme. |
| `speed` | `number` | `0.5` | Animation speed multiplier. Higher values make the animation faster. |
| `warpAmount` | `number` | `0` | Amount of wave distortion applied to the pattern (0-1). Creates a flowing, warped effect. |
| `resolutionScale` | `number` | `1` | Rendering resolution scale. Lower values (e.g., 0.5) improve performance on slower devices. |
| `className` | `string` | `undefined` | Additional CSS classes (note: width and height are always set to 100%). |

## Example
### Full Screen Background
```tsx
<div style={{ position: 'fixed', inset: 0, zIndex: -1 }}>
  <DarkVeil hueShift={120} speed={0.6} />
</div>
```

## How It Works

### Overview
DarkVeil renders animated neural network-generated patterns directly on the GPU using WebGL shaders. The visual output is created entirely procedurally—no images or textures are loaded.

### Architecture

#### 1. **CPPN (Compositional Pattern Producing Networks)**
The fragment shader contains a pre-trained neural network that generates patterns based on pixel coordinates:

- **Input Layer**: Takes pixel coordinates (x, y), distance from center, and time-based values
- **Hidden Layers**: 4 hidden layers with sigmoid activations containing pre-computed weights
- **Output Layer**: Produces RGB color values for each pixel

The network weights are hardcoded in the shader, making it extremely fast—no CPU computation needed.

#### 2. **WebGL Rendering Pipeline**

```
Canvas → Renderer → Program (Shaders) → Mesh → Output
```

- **Renderer**: OGL's WebGL renderer with device pixel ratio optimization
- **Vertex Shader**: Maps a fullscreen triangle to cover the entire canvas
- **Fragment Shader**: Runs the CPPN for every pixel in parallel on the GPU
- **Mesh**: A single triangle mesh (more efficient than a quad)

#### 3. **Visual Effects**

##### Hue Shift
Converts RGB → YIQ color space, rotates the IQ components (chrominance), then converts back to RGB. This allows smooth color transitions without breaking color harmony.

```glsl
// Simplified concept
YIQ = RGB × rgb2yiq_matrix
YIQ_shifted = rotate(YIQ.iq, hueShift)
RGB_output = YIQ_shifted × yiq2rgb_matrix
```

##### Wave Distortion (Warp)
Adds sine/cosine waves to UV coordinates before feeding them to the CPPN:

```glsl
uv += warpAmount * vec2(
  sin(uv.y * 2π + time * 0.5),
  cos(uv.x * 2π + time * 0.5)
) * 0.05
```

This creates flowing, organic motion in the pattern.

##### Animation
The CPPN receives time-modulated inputs that slowly evolve:
- `0.1 * sin(0.3 * time)` - slow oscillation
- `0.1 * sin(0.69 * time)` - medium oscillation  
- `0.1 * sin(0.44 * time)` - different phase oscillation

These create smooth, non-repeating transitions in the generated patterns.

#### 4. **Performance Optimizations**

- **Device Pixel Ratio Cap**: Limited to 2x to prevent excessive pixel processing on high-DPI displays
- **Resolution Scaling**: Renders at lower resolution then scales up via CSS
- **Single Triangle**: Uses one triangle instead of two (quad) to reduce vertex processing
- **GPU-Only**: All pattern generation happens in parallel on the GPU—no CPU bottleneck
- **Minimal State**: Only 4 uniforms updated per frame

### Data Flow

```
Props → React Hook → WebGL Uniforms → Shaders → GPU → Canvas
  ↓
Time → Animation Loop → Update Uniforms → Re-render
```

1. Component receives props (hueShift, speed, etc.)
2. `useDarkVeilGL` hook initializes WebGL context and shaders
3. Animation loop runs via `requestAnimationFrame`
4. Each frame updates uniform values (time, hue, warp)
5. GPU executes fragment shader for every pixel simultaneously
6. Result is drawn to canvas

### Why It's Fast

- **Parallel Processing**: Every pixel is computed simultaneously on the GPU
- **No Textures**: No image loading or texture sampling overhead
- **Minimal CPU**: Only updating 4 float values per frame
- **Compiled Shaders**: GLSL is compiled to native GPU instructions
- **Single Draw Call**: One triangle mesh = one draw call per frame

## Technical Details

- Built with [OGL](https://github.com/oframe/ogl) (Minimal WebGL library ~6KB)
- CPPN architecture: 6 inputs → 4 hidden layers (32 neurons) → 3 outputs (RGB)
- Shaders written in GLSL ES 1.0 for maximum compatibility
- Automatically handles canvas resizing and cleanup
- Optimized with device pixel ratio capping at 2x

## Performance Tips

- Set `resolutionScale` to `0.5` or `0.75` on mobile devices
- Lower `speed` values reduce GPU load slightly (mostly affects visual pacing)
- The component automatically cleans up WebGL resources on unmount
- On very old devices, consider wrapping in conditional rendering based on feature detection

