import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { NightLayer } from '../src/main';

const map = new maplibregl.Map({
  container: 'map', // container id
  style: 'https://demotiles.maplibre.org/style.json', // style URL
  center: [135, 0], // starting position [lng, lat]
  zoom: 0 // starting zoom
});

map.on('load', () => {
  map.addLayer(new NightLayer({
    color: [0, 0, 64],
    twilightSteps: 4,
  }));
});
