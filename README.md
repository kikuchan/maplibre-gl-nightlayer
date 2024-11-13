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
  map.addLayer(new NightLayer());
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

class NightLayer implements CustomLayerInterface {
    constructor(opts?: Options);

    getSubsolarPoint(date: Date): {
        subsolarLng: number;
        subsolarLat: number;
    };

    getDate(): Date | null;
    setDate(date: Date | null): void;

    getOpacity(): number;
    setOpacity(v: number): void;

    getColor(): Color;
    setColor(v: Color): void;

    getTwilightSteps(): number;
    setTwilightSteps(v: number): void;

    getTwilightStepAngle(): number;
    setTwilightStepAngle(v: number): void;

    getTwilightAttenuation(): number;
    setTwilightAttenuation(v: number): void;

      :
}
```
