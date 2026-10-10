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
    <div id="clock" class={() => (time() === null ? 'now' : '')} onclick={onclick}>
      <div class="day">{() => `${months[Number(d().month()) - 1]} ${d().day()}`}</div>
      <div class="ym">{() => d().format('g')}</div>
      <div class="hm">{hm}</div>
    </div>
  );
}

function App() {
  const [time, setTime] = createSignal<Decimal | null>(null);
  const [timeAnimating, setTimeAnimating] = createSignal<Decimal | null>(null);
  const [zoom, setZoom] = createSignal<number>(-8);

  const [twilights, setTwilights] = createSignal(4);
  const [visible, setVisible] = createSignal(true);
  const [flat, setFlat] = createSignal(false);

  const [animating, setAnimating] = createSignal(false);
  const [editing, setEditing] = createSignal(false);
  const [now, setNow] = createSignal(Date.now());

  onMount(() => {
    const map = new maplibregl.Map({
      container: 'map',
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: ['https://a.tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
            attribution: '&copy; OpenStreetMap Contributors',
            maxzoom: 19,
          },
        },
        layers: [
          {
            id: 'osm',
            type: 'raster',
            source: 'osm',
          },
        ],
      },
      center: [135, 35],
      zoom: 4,
      attributionControl: false,
    });
    map.addControl(new maplibregl.NavigationControl());
    const attr = new maplibregl.AttributionControl({
      compact: true,
    });
    map.addControl(attr);

    const nightLayer = new NightLayer({
      color: [0, 0, 64],
      twilightSteps: 0,
    });

    const [ready, setReady] = createSignal(false);
    map.on('load', () => {
      map.addLayer(nightLayer);
      setReady(true);
      attr._toggleAttribution();
    });

    effect(() => {
      nightLayer.setTwilightSteps(twilights());
    });

    effect(() => {
      nightLayer.setOpacity(visible() ? 0.5 : 0);
    });

    effect(() => {
      const t = timeAnimating();
      nightLayer.setDate(t === null ? null : new Date(t.mul(1000).number()));
    });

    effect(() => {
      const type = flat() ? 'mercator' : 'globe';
      if (!ready()) return;
      map.setProjection({ type });
    });
  });

  setInterval(() => {
    if (animating() || editing()) return;
    setNow(Date.now());
  }, 200);

  return (
    <>
      <div id="map"></div>
      <div
        style="position: fixed; left: 1em; top: 1em; display: flex; gap: 0.25rem; right: 1em; align-items: center;"
        class="popup"
      >
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
          <button onclick={() => setZoom((z) => z + 1)}>+</button>
          <button onclick={() => setZoom((z) => z - 1)}>-</button>
        </div>
      </div>
      <div style="position: fixed; left: 1em; bottom: 1em; display: flex; gap: 0.5em; flex-wrap: wrap;" class="popup">
        <div>
          Shadow{' '}
          <button id="btn-nightlayer" style="white-space: nowrap" onclick={() => setVisible((v) => !v)}>
            → {() => (visible() ? 'OFF' : 'ON')}
          </button>
        </div>
        <div>
          Twilights{' '}
          <select
            value={twilights}
            onchange={(e: Event) => setTwilights(Number((e.currentTarget as HTMLSelectElement).value))}
            style="white-space: nowrap"
          >
            <option value="0">Gradation</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4" selected>4</option>
          </select>
        </div>
        <div>
          Map{' '}
          <button id="btn-projection" style="white-space: nowrap" onclick={() => setFlat((v) => !v)}>
            → {() => (flat() ? 'Globe' : 'Flat Earth')}
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
