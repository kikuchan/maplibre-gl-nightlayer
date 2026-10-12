import { createMemo, createSignal, effect, onMount, render } from '@luna_ui/luna';
import * as maplibregl from 'maplibre-gl';
import maplibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
import 'maplibre-gl/dist/maplibre-gl.css';
import './main.css';
import { NightLayer } from '../../src/main';
import { Timescope } from '@timescope/luna';
import { Decimal, Calendar } from 'timescope';

maplibregl.setWorkerUrl(maplibreWorkerUrl);

const DATE_RANGE_SECONDS = 8_640_000_000_000;
type Basemap = 'satellite' | 'osm';

function Clock({ time, now, onclick }: { time: () => Decimal | null; now: () => number; onclick: () => void }) {
  const d = createMemo(() => Calendar.fromEpoch(time() ?? Decimal(now()).div(1000)));
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const hm = createMemo(() => {
    const calendar = d();
    const hour = calendar.hour();
    const hour12 = (hour % 12n || 12n).toString().padStart(2, '0');
    return `${hour12}:${calendar.format('mm')} ${hour < 12n ? 'AM' : 'PM'}`;
  });

  return (
    <button
      id="clock"
      type="button"
      aria-label="Return to current time"
      class={() => (time() === null ? 'now' : '')}
      onclick={onclick}
    >
      <span class="day">{() => `${months[Number(d().month()) - 1]} ${d().day()}`}</span>
      <span class="ym">{() => d().format('g')}</span>
      <span class="hm">{hm}</span>
    </button>
  );
}

function App() {
  const query = new URLSearchParams(location.search);
  const numberParam = (name: string, fallback: number, min = -Infinity, max = Infinity) => {
    const text = query.get(name);
    const value = text === null || text.trim() === '' ? NaN : Number(text);
    return Number.isFinite(value) && value >= min && value <= max ? value : fallback;
  };
  const date = numberParam('date', NaN, -DATE_RANGE_SECONDS, DATE_RANGE_SECONDS);
  const initialTime = Number.isFinite(date) ? Decimal(date) : null;
  const center = query.get('center')?.split(',').map(Number);
  const initialCenter: [number, number] =
    center?.length === 2 && center.every(Number.isFinite) && Math.abs(center[1]) <= 90
      ? [center[0], center[1]]
      : [135, 35];
  const [time, setTime] = createSignal<Decimal | null>(initialTime);
  const [timeAnimating, setTimeAnimating] = createSignal<Decimal | null>(initialTime);
  const [zoom, setZoom] = createSignal(numberParam('timelineZoom', -8, -35, 10));

  const twilightSteps = numberParam('twilights', 0, 0, 4);
  const [twilights, setTwilights] = createSignal(Number.isInteger(twilightSteps) ? twilightSteps : 0);
  const [opacity, setOpacity] = createSignal(numberParam('opacity', 1, 0, 1));
  const [globe, setGlobe] = createSignal(query.get('projection') !== 'mercator');
  const [basemap, setBasemap] = createSignal<Basemap>(query.get('basemap') === 'osm' ? 'osm' : 'satellite');
  const [atmosphere, setAtmosphere] = createSignal(numberParam('atmosphere', 1, 0, 1));
  const [terrain, setTerrain] = createSignal(query.get('terrain') === 'true');

  const [animating, setAnimating] = createSignal(false);
  const [editing, setEditing] = createSignal(false);
  const [now, setNow] = createSignal(Date.now());

  onMount(() => {
    const map = new maplibregl.Map({
      container: 'map',
      style: {
        version: 8,
        sources: {
          satellite: {
            type: 'raster',
            tiles: ['https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg'],
            tileSize: 256,
            attribution:
              '<a href="https://cloudless.eox.at/">Sentinel-2 cloudless by EOX</a> ' +
              '(Contains modified Copernicus Sentinel data 2020) ' +
              '<a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>',
            maxzoom: 19,
          },
          osm: {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
            maxzoom: 19,
          },
          terrainSource: {
            type: 'raster-dem',
            url: 'https://tiles.mapterhorn.com/tilejson.json',
          },
        },
        layers: [
          {
            id: 'satellite',
            type: 'raster',
            source: 'satellite',
            layout: { visibility: basemap() === 'satellite' ? 'visible' : 'none' },
          },
          {
            id: 'osm',
            type: 'raster',
            source: 'osm',
            layout: { visibility: basemap() === 'osm' ? 'visible' : 'none' },
          },
        ],
        projection: {
          type: globe() ? 'globe' : 'mercator',
        },
      },
      center: initialCenter,
      zoom: numberParam('zoom', 1.5, 0, 22),
      pitch: numberParam('pitch', 0, 0, 85),
      bearing: numberParam('bearing', 0),
      roll: numberParam('roll', 0),
      maxPitch: 85,
      attributionControl: false,
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }));
    const attr = new maplibregl.AttributionControl({
      compact: false,
    });
    map.addControl(attr);

    const nightLayer = new NightLayer({
      opacity: opacity(),
      atmosphere: atmosphere(),
      updateInterval: 10000,
    });

    let urlTimer: ReturnType<typeof setTimeout>;
    const refreshURL = () => {
      clearTimeout(urlTimer);
      urlTimer = setTimeout(() => {
        const url = new URL(location.href);
        const center = map.getCenter();
        const values = {
          center: `${center.lng.toFixed(8)},${center.lat.toFixed(8)}`,
          zoom: map.getZoom().toFixed(3),
          pitch: map.getPitch().toFixed(3),
          bearing: map.getBearing().toFixed(3),
          roll: map.getRoll().toFixed(3),
          projection: globe() ? 'globe' : 'mercator',
          basemap: basemap(),
          terrain: terrain(),
          atmosphere: atmosphere().toFixed(3),
          opacity: opacity().toFixed(3),
          twilights: twilights().toFixed(0),
          timelineZoom: zoom().toFixed(2),
        };
        for (const [key, value] of Object.entries(values)) url.searchParams.set(key, String(value));
        const date = timeAnimating();
        if (date === null) url.searchParams.delete('date');
        else url.searchParams.set('date', date.toFixed(0));
        if (url.href !== location.href) history.replaceState(history.state, '', url);
      }, 200);
    };
    map.on('move', refreshURL);
    effect(() => {
      globe();
      basemap();
      terrain();
      atmosphere();
      opacity();
      twilights();
      timeAnimating();
      zoom();
      refreshURL();
    });

    const [ready, setReady] = createSignal(false);
    map.on('load', () => {
      map.addLayer(nightLayer);
      setReady(true);
    });

    effect(() => {
      nightLayer.setTwilightSteps(twilights());
    });

    effect(() => {
      nightLayer.setOpacity(opacity());
    });

    effect(() => {
      const t = timeAnimating();
      nightLayer.setDate(t === null ? null : new Date(t.mul(1000).number()));
    });

    effect(() => {
      const type = globe() ? 'globe' : 'mercator';
      if (!ready()) return;
      map.setProjection({ type });
    });

    effect(() => {
      const selected = basemap();
      if (!ready()) return;
      map.setLayoutProperty('satellite', 'visibility', selected === 'satellite' ? 'visible' : 'none');
      map.setLayoutProperty('osm', 'visibility', selected === 'osm' ? 'visible' : 'none');
    });

    effect(() => {
      if (!ready()) return;
      nightLayer.setAtmosphere(atmosphere());
      map.setSky({
        'sky-color': 'transparent',
        'horizon-color': 'transparent',
        'fog-color': 'transparent',
        'fog-ground-blend': 1,
        'atmosphere-blend': 0,
      });
    });
    effect(() => {
      if (!ready()) return;
      map.setTerrain(terrain() ? { source: 'terrainSource', exaggeration: 1 } : null);
    });
  });

  setInterval(() => {
    if (animating() || editing()) return;
    setNow(Date.now());
  }, 200);

  return (
    <>
      <div id="map"></div>
      <div class="popup time-panel">
        <Clock time={timeAnimating} now={now} onclick={() => setTime(null)} />
        <Timescope
          time={time}
          zoom={zoom}
          style="height: 48px; background: transparent;"
          timeRange={[-DATE_RANGE_SECONDS, DATE_RANGE_SECONDS]}
          zoomRange={[-35, 10]}
          onTimeChanged={setTime}
          onZoomChanged={setZoom}
          onTimeAnimating={(value) => {
            setTimeAnimating(value?.clamp(-DATE_RANGE_SECONDS, DATE_RANGE_SECONDS) ?? null);
          }}
          onAnimating={setAnimating}
          onEditing={setEditing}

          options={{
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
          }}
        />
        <div class="controls">
          <button aria-label="Zoom in on timeline" onclick={() => setZoom((z) => z + 1)}>
            +
          </button>
          <button aria-label="Zoom out on timeline" onclick={() => setZoom((z) => z - 1)}>
            -
          </button>
        </div>
      </div>
      <div class="popup display-controls">
        <div>
          <label for="atmosphere-blend">Atmosphere</label>
          <input
            id="atmosphere-blend"
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={atmosphere}
            oninput={(e: Event) => setAtmosphere(Number((e.currentTarget as HTMLInputElement).value))}
          />
          <output for="atmosphere-blend">{() => atmosphere().toFixed(2)}</output>
        </div>
        <div>
          <label for="layer-opacity">Layer opacity</label>
          <input
            id="layer-opacity"
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={opacity}
            oninput={(e: Event) => setOpacity(Number((e.currentTarget as HTMLInputElement).value))}
          />
          <output for="layer-opacity">{() => opacity().toFixed(2)}</output>
        </div>
        <div>
          <label for="twilight-steps">Twilights</label>
          <select
            id="twilight-steps"
            value={twilights}
            ref={(select) =>
              queueMicrotask(() => {
                (select as HTMLSelectElement).value = String(twilights());
              })
            }
            onchange={(e: Event) => setTwilights(Number((e.currentTarget as HTMLSelectElement).value))}
          >
            <option value="0">Gradation</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4">4</option>
          </select>
        </div>
        <div>
          <label for="basemap">Map</label>
          <select
            id="basemap"
            value={basemap}
            ref={(select) =>
              queueMicrotask(() => {
                (select as HTMLSelectElement).value = basemap();
              })
            }
            onchange={(e: Event) => setBasemap((e.currentTarget as HTMLSelectElement).value as Basemap)}
          >
            <option value="satellite">Satellite</option>
            <option value="osm">OpenStreetMap</option>
          </select>
        </div>
        <div>
          <span id="globe-label">Globe</span>
          <button
            id="btn-projection"
            type="button"
            class="toggle-switch"
            role="switch"
            aria-labelledby="globe-label"
            aria-checked={globe}
            onclick={() => setGlobe((v) => !v)}
          >
            <span class="toggle-track" aria-hidden="true" />
          </button>
        </div>
        <div>
          <span id="terrain-label">Terrain</span>
          <button
            id="btn-terrain"
            type="button"
            class="toggle-switch"
            role="switch"
            aria-labelledby="terrain-label"
            aria-checked={terrain}
            onclick={() => setTerrain((v) => !v)}
          >
            <span class="toggle-track" aria-hidden="true" />
          </button>
        </div>
      </div>
    </>
  );
}

const app = document.getElementById('app');
if (app) {
  render(app, <App />);
}
