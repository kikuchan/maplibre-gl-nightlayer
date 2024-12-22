# A Night Layer for MapLibre GL JS

Adds a simple night shadow (Earth's shadow) on the map.
![image](https://github.com/user-attachments/assets/425f58e0-66f6-4779-8c03-ff543972c59e)

## Install
```sh
$ npm install --save-dev maplibre-gl-nightlayer
```

## Usage

```js
import { NightLayer } from 'maplibre-gl-nightlayer';

  :

map.on('load', () => {
  map.addLayer(new NightLayer({
    // These are the default values
    date: null,
    opacity: 0.5,
    color: [0, 0, 0, 255],
    daytimeColor: [0, 0, 0, 0], // transparent
    twilightSteps: 0,
    twilightAttenuation: 0.5,
    updateInterval: 10000, // in milliseconds
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

    :
};

export function getSubsolarPoint(date?: Date): {
  lng: number;
  lat: number;
};

```
