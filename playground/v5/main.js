import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { NightLayer } from '../../src/main';

const map = new maplibregl.Map({
  container: 'map', // container id
  style: 'https://demotiles.maplibre.org/style.json', // style URL
  center: [135, 0], // starting position [lng, lat]
  zoom: 0, // starting zoom
});

map.on('load', () => {
  const nightLayer = new NightLayer({
    color: [0, 0, 64],
    twilightSteps: 0,
  });
  map.addLayer(nightLayer);

  document.getElementById('btn-projection').addEventListener('click', () => {
    const p = map.style.projection.name === 'globe' ? 'mercator' : 'globe';
    map.setProjection({ type: p });
  });

  document.getElementById('btn-nightlayer').addEventListener('click', () => {
    const v = nightLayer.getOpacity() > 0 ? 0 : 0.5;
    nightLayer.setOpacity(v);
  });

  document.getElementById('sel-twilightstep').addEventListener('change', (e) => {
    nightLayer.setTwilightSteps(e.target.value);
  });
});
