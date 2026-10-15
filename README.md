# A Night Layer for MapLibre GL JS

Adds a simple night shadow (Earth's shadow) on the map.
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

    :
};

export function getSubsolarPoint(date?: Date | null): {
  lng: number;
  lat: number;
};

```

The default LUT is embedded. Use `lut` to load a custom LUT ZIP by URL. Requires `OES_texture_float_linear`.
