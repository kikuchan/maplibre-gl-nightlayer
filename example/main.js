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
  center: [135, 35], // starting position [lng, lat]
  zoom: 4, // starting zoom
  attributionControl: false,
});
map.addControl(new maplibregl.NavigationControl());
const attr = new maplibregl.AttributionControl({
  compact: true,
})
map.addControl(attr);


const nightLayer = new NightLayer({
  color: [0, 0, 64],
  twilightSteps: 0,
});

const timescope = new Timescope({
  target: '#timescope',
  zoom: -8,
  timeRange: [undefined, undefined],
  tracks: {
    default: {
      timeAxis: {
        axis: {
          color: 'white',
        },
        ticks: {
          color: 'white',
        },
        labels: {
          color: 'white',
        },
      },
    },
  },
  style: {
    height: '48px',
    background: 'transparent',
  },
});

map.on('load', () => {
  map.addLayer(nightLayer);
  map.setProjection({ type: 'globe' });
  attr._toggleAttribution();

  document.getElementById('btn-projection').addEventListener('click', () => {
    const p = map.style.projection.name === 'globe' ? 'mercator' : 'globe';
    map.setProjection({ type: p });
    document.getElementById('btn-projection').textContent = p === 'mercator' ? `→ Globe` : `→ Flat Earth`;
  });

  document.getElementById('btn-nightlayer').addEventListener('click', () => {
    const v = nightLayer.getOpacity() > 0 ? 0 : 0.5;
    nightLayer.setOpacity(v);

    document.getElementById('btn-nightlayer').textContent = v ? `→ OFF` : `→ ON`;
  });

  nightLayer.setTwilightSteps(4);
  document.getElementById('sel-twilightstep').addEventListener('change', (e) => {
    nightLayer.setTwilightSteps(e.target.value);
  });


  document.getElementById('btn-timescope-zoom-in').addEventListener('click', () => {
    timescope.setZoom(timescope.zoom + 1);
  });
  document.getElementById('btn-timescope-zoom-out').addEventListener('click', () => {
    timescope.setZoom(timescope.zoom - 1);
  });
  document.getElementById('clock').addEventListener('click', () => {
    timescope.setTime(null);
  });

  showClock(Date.now());
});

function showClock(t) {
  const clockEl = document.getElementById('clock');

  if (isFinite(t)) {
    const d = new Date(t);
    const m = new Intl.DateTimeFormat("en-US", {
      'month': 'short',
      'day': 'numeric',
    });
    const y = new Intl.DateTimeFormat("en-US", {
      'year': 'numeric',
    });
    const hm = new Intl.DateTimeFormat("en-US", {
      'hour': '2-digit',
      'minute': '2-digit',
    });

    if (clockEl) {
      clockEl.innerHTML = `
        <div class="day">${m.format(d)}</div>
        <div class="ym">${y.format(d)}</div>
        <div class="hm">${hm.format(d)}</div>
      `;
    }
  } else {
  }

  clockEl.style.color = timescope.time === null && !timescope.animating && !timescope.editing ? 'white' : 'yellow';
}

timescope.on('timeanimating', (e) => {
  nightLayer.setTime(e.value)

  const t = e.value?.mul(1000).number() || Date.now();
  showClock(t);
});

setInterval(() => {
  const t = timescope.time?.mul(1000).number() || Date.now();
  if (!timescope.animating && !timescope.editing) {
    showClock(t);
  }
}, 200);
