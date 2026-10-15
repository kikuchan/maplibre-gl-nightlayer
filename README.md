# A Night Layer for MapLibre GL JS

Adds a simple night shadow (Earth's shadow) on the map.

> [!WARNING]
> This package relies on MapLibre GL JS internals. Compatibility may break even in minor or patch releases.

![image](https://github.com/user-attachments/assets/425f58e0-66f6-4779-8c03-ff543972c59e)

## Demo
[example](https://kikuchan.github.io/maplibre-gl-nightlayer/)

## Usage
### Module
```sh
$ npm install --save-dev maplibre-gl-nightlayer
```
```js
import { NightLayer } from 'maplibre-gl-nightlayer';

  :

map.on('load', () => {
  map.addLayer(new NightLayer({
    date: null,
    opacity: 0.5,
    color: [0, 0, 0, 0],
    daytimeColor: [0, 0, 0, 0],
    twilightSteps: 0,
    twilightAttenuation: 1,
    updateInterval: 10000,
    atmosphere: 0,
    exposure: 0,
    autoExposure: false,
    stars: true,
  }));
});
```

### via CDN

Load MapLibre GL JS first so that `maplibregl` is available globally.

```html
<script src="https://unpkg.com/maplibre-gl-nightlayer@latest/dist/nightlayer.min.js"></script>
```
```js
map.on('load', () => {
  map.addLayer(new nightLayer.NightLayer({
    :
  }));
});
```

## API
```ts
type Color3 = [number, number, number];
type Color4 = [number, number, number, number];
type Color = Color3 | Color4;

type Options = {
  date?: Date | null;
  opacity?: number;
  color?: Color;
  daytimeColor?: Color;
  twilightSteps?: number;
  twilightAttenuation?: number;
  updateInterval?: number;
  atmosphere?: number;
  exposure?: number;
  autoExposure?: boolean;
  stars?: boolean;
  lut?: string | URL;
};

export class NightLayer implements CustomLayerInterface {
  constructor(opts?: Options);

  getSubsolarPoint(): {
    lng: number;
    lat: number;
  };

  getDate(): Date | null;
  setDate(date: Date | null): void;

  getOpacity(): number;
  setOpacity(opacity: number): void;

  getColor(): Color4;
  setColor(color: Color): void;

  getDaytimeColor(): Color4;
  setDaytimeColor(color: Color): void;

  getTwilightSteps(): number;
  setTwilightSteps(steps: number): void;

  getTwilightAttenuation(): number;
  setTwilightAttenuation(attenuation: number): void;

  getUpdateInterval(): number;
  setUpdateInterval(interval: number): void;

  getAtmosphere(): number;
  setAtmosphere(atmosphere: number): void;

  getExposure(): number;
  setExposure(exposure: number): void;

  getAutoExposure(): boolean;
  setAutoExposure(enabled: boolean): void;

  getStars(): boolean;
  setStars(enabled: boolean): void;

    :
};

export function getSubsolarPoint(date?: Date | null): {
  lng: number;
  lat: number;
};

```

The default LUT is embedded. Use `lut` to load a custom LUT ZIP by URL. Requires `OES_texture_float_linear`.

`exposure` adjusts brightness before tone mapping, from -12 to +24 stops (default: 0). Each +1 stop doubles the light.

`autoExposure` enables gradual, center-weighted scene metering (default: false). `exposure` then acts as compensation; automatic adjustment is limited to -4 through +24 stops.

`stars` shows 8,920 catalogue stars through V=6.5 and a diffuse Milky Way (default: true). Point-star flux is tied to the Sun's V=-26.74 magnitude, with approximate B-V-based colours and V-to-luminance conversion. Stars remain point-like when zooming and share atmospheric extinction and exposure. Night skies require much more exposure than daylight.

Point stars are rasterized as one-vertex point sprites into a cached half-float flux target, avoiding catalogue searches at every sky pixel. Edge-clamped sprites preserve partially visible stars at screen boundaries; devices with insufficient point-size support use instanced quads instead. The final composite reads this target once and applies the shared atmospheric extinction and exposure. This adds one sparse star pass and an 8-byte-per-pixel working target (about 16 MiB at 1920×1080); the map still needs only one framebuffer copy and one final composite. Fixed-date views reuse the target until the viewing transform or size changes.

The selected date also controls the Moon's position, phase and distance. Moonlight lights the ground and scatters through the same atmospheric LUTs, including the glow near the Moon. Full-Moon normal illuminance is approximately 0.25 lux before atmospheric extinction, with an empirical phase law (a half Moon is roughly one tenth as bright). The disk uses an analytic phase shape, not a lunar surface texture. Eclipses, clouds, ice-crystal halos and lens bloom are not modeled; positions use a low-precision ephemeris, not an occultation-grade calculation.

Ground lighting includes a uniform natural-night floor of 0.0008 lux on an unobstructed horizontal surface, representing stars, the Milky Way, airglow and zodiacal light together ([NPS reference](https://www.nps.gov/subjects/nightskies/NSQmetrics.html)). It does not follow the Milky Way's orientation or add an airglow image to the sky. In vacuum, only a rough 25% extraterrestrial-star share of this floor remains; direct moonlight and the lunar disk remain, while lunar scattering is disabled. These lights apply to continuous illumination; stepped twilight categories still depend only on solar altitude. `stars` controls the visible starfield, not this aggregate ground-light floor.

Star data includes proper motion within 1800–2200; dates outside that range clamp proper motion. The 1K Milky Way background is empirically normalized from unsaturated stars (roughly ±0.2 magnitude scatter), not precision photometry, and the sky is not a complete faint-star catalogue. Credits: HYG / David Nash ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)), [NASA/GSFC/SVS](https://svs.gsfc.nasa.gov/4851/), ESA/Gaia/DPAC; see `THIRD-PARTY-NOTICES`.
