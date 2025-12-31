import { createMemo, createSignal, effect, onMount, render } from "@luna_ui/luna";
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import './main.css';
import { NightLayer } from '../../src/main';
import { Decimal, Timescope } from '@timescope/luna';

function Clock({ time, now, onclick }: any) {
    const d = createMemo(() => new Date(time()?.mul(1000).number() ?? now()));
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

  return (<div id="clock" class={() => time() === null ? 'now' : ''} onclick={onclick}>
    <div class="day">{() => m.format(d())}</div>
    <div class="ym">{() => y.format(d())}</div>
    <div class="hm">{() => hm.format(d())}</div>
  </div>);
}

function App() {
  const [time, setTime] = createSignal<Decimal | null>(null);
  const [timeAnimating, setTimeAnimating] = createSignal<Decimal | null>(null);
  const [zoom, setZoom] = createSignal<number>(-8);

  const [twilights, setTwilights] = createSignal(4);
  const [visible, setVisible] = createSignal(true);
  const [flat, setFlat] = createSignal(false);

  onMount(() => {
    const map = new maplibregl.Map({
      container: 'map', // container id
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "&copy; OpenStreetMap Contributors",
            maxzoom: 19
          }
        },
        layers: [
          {
            id: "osm",
            type: "raster",
            source: "osm"
          }
        ]
      },
      center: [135, 35],
      zoom: 4,
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

    let ready = false;
    map.on('load', () => {
      ready = true;
      map.addLayer(nightLayer);
      map.setProjection({ type: 'globe' });
      attr._toggleAttribution();
    });

    effect(() => {
      nightLayer.setTwilightSteps(twilights());
    });

    effect(() => {
      nightLayer.setOpacity(visible() ? 0.5 : 0);
    });

    effect(() => {
      const type = flat() ? 'mercator' : 'globe';
      if (!ready) return;
      map.setProjection({ type });
    });
  });

  const [animating, setAnimating] = createSignal(false);
  const [editing, setEditing] = createSignal(false);
  const [now, setNow] = createSignal(Date.now());

  setInterval(() => {
    if (animating() || editing()) return;
    setNow(Date.now());
  }, 200);

  return (
    <>
      <div id="map"></div>
      <div style="position: fixed; left: 1em; top: 1em; display: flex; gap: 0.25rem; right: 1em; align-items: center;" class="popup">
        <Clock time={timeAnimating} now={now} onclick={() => setTime(null)} />
        <Timescope
          time={time}
          zoom={zoom}
          height={() => '48px'}
          timeRange={() => [undefined, undefined]}
          onTimeChanged={setTime}
          onTimeAnimating={setTimeAnimating}
          onAnimating={setAnimating}
          onEditing={setEditing}

          background={() => 'transparent'}
          tracks={() => ({
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
            }
          },
        })}/>
        <div class="controls">
          <button onclick={() => setZoom((z) => z + 1)}>+</button>
          <button onclick={() => setZoom((z) => z - 1)}>-</button>
        </div>
      </div>
      <div style="position: fixed; left: 1em; bottom: 1em; display: flex; gap: 0.5em; flex-wrap: wrap;" class="popup">
        <div>Shadow <button id="btn-nightlayer" style="white-space: nowrap" onclick={() => setVisible((v) => !v)}>→ {() => visible() ? 'OFF' : 'ON'}</button></div>
        <div>
          Twilights <select onchange={(e: any) => setTwilights(Number(e.target.value))} style="white-space: nowrap">
            <option value="0">Gradation</option>
            <option value="1">1</option>
            <option value="2">2</option>
            <option value="3">3</option>
            <option value="4" selected>4</option>
          </select>
        </div>
        <div>
        Map <button id="btn-projection" style="white-space: nowrap" onclick={() => setFlat((v) => !v)}>→ {() => flat() ? 'Globe' : 'Flat Earth'}</button>
        </div>
      </div>
    </>
  );
}

const app = document.getElementById("app");
if (app) {
  render(app, <App />);
}

