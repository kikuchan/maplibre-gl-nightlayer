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
    color: [0, 0, 0],
    twilightSteps: 0,
    twilightStepAngle: 6,
    twilightAttenuation: 0.5,
  }));
});
```

## API
```ts
type Color = [number, number, number];

type Options = {
  date?: Date | null;
  opacity?: number;
  color?: Color;
  twilightSteps?: number;
  twilightStepAngle?: number;
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

  getColor(): Color;
  setColor(color: Color): void;

  getTwilightSteps(): number;
  setTwilightSteps(steps: number): void;

  getTwilightStepAngle(): number;
  setTwilightStepAngle(angle: number): void;

  getTwilightAttenuation(): number;
  setTwilightAttenuation(attenuation: number): void;

    :
};

export function getSubsolarPoint(date?: Date): {
  lng: number;
  lat: number;
};

```
