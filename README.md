# A Night Layer for MapLibre GL JS

Night falls on your map: twilight, starlight and moonlight.

> [!WARNING]
> This package relies on MapLibre GL JS internals. Compatibility may break even in minor or patch releases.

<img width="1680" height="1280" alt="image" src="https://github.com/user-attachments/assets/ed8f5264-f997-4b3c-990c-35a2454a4e91" />

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
    twilight: null,
    updateInterval: 10000,
    atmosphere: false,
    development: false,
  }));
});
```

### via CDN

```html
<script type="module">
  import { NightLayer } from 'https://unpkg.com/maplibre-gl-nightlayer';

    :

  map.on('load', () => {
    map.addLayer(new NightLayer());
  });
</script>
```

## API

```ts
type Color3 = [number, number, number];
type Color4 = [number, number, number, number];
type Color = Color3 | Color4;
type TwilightSteps = 1 | 2 | 3 | 4;
type Twilight = { steps: TwilightSteps; color: Color4; attenuation: number };
type TwilightInput = TwilightSteps | {
  steps: TwilightSteps;
  color?: Color; // RGB/alpha 0..1; default [0,0,0,0.95]; RGB-only alpha defaults to 0.95
  attenuation?: number; // positive, default 0.5
} | null;
type DevelopmentProcess = 'linear' | 'aces' | 'agx' | 'filmic';
type Development = {
  process: DevelopmentProcess;
  exposure: number;        // -12..32 EV
  autoExposure: boolean;
  compensation: number;    // -12..12 EV, AE only
  gamma: number;           // 0.1..4, post-development
};
type DevelopmentInput = false | DevelopmentProcess | Partial<Development>;
type Atmosphere = { sky: number; fog: number; stars: boolean };
type AtmosphereInput = boolean | number | {
  sky?: boolean | number;
  fog?: boolean | number;
  stars?: boolean;
};

type Options = {
  date?: Date | null;
  opacity?: number;
  twilight?: TwilightInput; // map-only stepped lighting; null/omission keeps continuous lighting
  updateInterval?: number;
  atmosphere?: AtmosphereInput; // default false
  development?: DevelopmentInput; // default false
  lut?: string;
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

  getTwilight(): Twilight | null;
  setTwilight(twilight?: TwilightInput): void;

  getUpdateInterval(): number;
  setUpdateInterval(interval: number): void;

  getAtmosphere(): Atmosphere;
  setAtmosphere(atmosphere: AtmosphereInput): void;


  getDevelopment(): Development | false;
  setDevelopment(development: DevelopmentInput): void;
  setLUT(source?: string): Promise<void>;
  captureEXR(options?: { exposure?: 'preview' | 'raw' }): Promise<Blob>;

    :
};

export function getSubsolarPoint(date?: Date | null): {
  lng: number;
  lat: number;
};

```
