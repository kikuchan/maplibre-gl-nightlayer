import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { NightLayer } from '../src/main';
import { Timescope } from 'timescope';

const map = new maplibregl.Map({
  container: 'map', // container id
  //style: 'https://demotiles.maplibre.org/style.json', // style URL
  style: {
    "version": 8,
    "sources": {
      "osm": {
        "type": "raster",
        "tiles": ["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"],
        "tileSize": 256,
        "attribution": "&copy; OpenStreetMap Contributors",
        "maxzoom": 19
      }
    },
    "layers": [
      {
        "id": "osm",
        "type": "raster",
        "source": "osm"
      }
    ]
  },
  center: [135, 0], // starting position [lng, lat]
  zoom: 0, // starting zoom
});

const nightLayer = new NightLayer({
  color: [0, 0, 64],
  twilightSteps: 0,
});

const timescope = new Timescope({
  target: '#timescope',
  zoom: -14,
  timeRange: [undefined, undefined],
  tracks: {
    default: {
      symmetric: true,
    },
  },
  style: {
    height: '48px'
  },
});

map.on('load', () => {
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

timescope.on('timeanimating', (e) => nightLayer.setTime(e.value));
