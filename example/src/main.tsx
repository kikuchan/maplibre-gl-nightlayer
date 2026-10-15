import { createMemo, createSignal, effect, onMount, render } from '@luna_ui/luna';
import * as maplibregl from 'maplibre-gl';
import maplibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
import 'maplibre-gl/dist/maplibre-gl.css';
import './main.css';
import { NightLayer } from '../../src/main';
import { LookAround } from './look-around';
import { DEFAULT_FOV, MAX_FOV, MIN_FOV } from './camera-settings';
import { Timescope } from '@timescope/luna';
import { Decimal, Calendar } from 'timescope';

maplibregl.setWorkerUrl(maplibreWorkerUrl);

const DATE_RANGE_SECONDS = 8_640_000_000_000;
const SETTINGS_ICON = `data:image/svg+xml,${encodeURIComponent(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#f1f5f9" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="m9.5 3-.6 2.1-1.6.9-2.1-.5-2.5 4.3 1.5 1.6v1.8l-1.5 1.6 2.5 4.3 2.1-.5 1.6.9.6 2.1h5l.6-2.1 1.6-.9 2.1.5 2.5-4.3-1.5-1.6v-1.8l1.5-1.6-2.5-4.3-2.1.5-1.6-.9L14.5 3Z"/>
    <circle cx="12" cy="12" r="3"/>
  </svg>
`)}`;
const LOOK_AROUND_ICON = `data:image/svg+xml,${encodeURIComponent(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#f1f5f9" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z"/>
    <circle cx="12" cy="12" r="3"/>
  </svg>
`)}`;
const RESTORE_ICON = `data:image/svg+xml,${encodeURIComponent(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#f1f5f9" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 10a9 9 0 1 1 2.6 8.4M3 4v6h6"/>
  </svg>
`)}`;
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
  const [exposure, setExposure] = createSignal(numberParam('exposure', 0, -12, 12));
  const [autoExposure, setAutoExposure] = createSignal(query.get('autoExposure') === 'true');
  const [terrain, setTerrain] = createSignal(query.get('terrain') === 'true');
  const [canLookAround, setCanLookAround] = createSignal(false);
  const [lookingAround, setLookingAround] = createSignal(false);
  const [fieldOfView, setFieldOfView] = createSignal(numberParam('fov', DEFAULT_FOV, MIN_FOV, MAX_FOV));

  const [animating, setAnimating] = createSignal(false);
  const [editing, setEditing] = createSignal(false);
  const [now, setNow] = createSignal(Date.now());
  let copyView = async () => {};
  let loadView = () => {};
  let toggleLookAround = () => {};
  let changeFieldOfView = (_value: number) => {};
  let fovDrag: { id: number; x: number } | undefined;
  const startFovDrag = (event: Event) => {
    const pointer = event as PointerEvent;
    if (fovDrag || pointer.button !== 0) return;
    event.preventDefault();
    const slider = event.currentTarget as HTMLInputElement;
    slider.focus({ preventScroll: true });
    slider.setPointerCapture(pointer.pointerId);
    fovDrag = { id: pointer.pointerId, x: pointer.clientX };
  };
  const moveFovDrag = (event: Event) => {
    const pointer = event as PointerEvent;
    if (fovDrag?.id !== pointer.pointerId) return;
    event.preventDefault();
    const slider = event.currentTarget as HTMLInputElement;
    // Eight slider-widths span the full range, allowing fine adjustments even
    // on a compact control. Native keyboard input still uses the small step.
    const delta = (pointer.clientX - fovDrag.x) / slider.clientWidth;
    const log = Math.log10(fieldOfView()) + (delta * Math.log10(MAX_FOV / MIN_FOV)) / 8;
    changeFieldOfView(Math.max(MIN_FOV, Math.min(MAX_FOV, 10 ** log)));
    fovDrag.x = pointer.clientX;
  };
  const endFovDrag = (event: Event) => {
    const pointer = event as PointerEvent;
    if (fovDrag?.id !== pointer.pointerId) return;
    fovDrag = undefined;
    const slider = event.currentTarget as HTMLInputElement;
    if (slider.hasPointerCapture(pointer.pointerId)) slider.releasePointerCapture(pointer.pointerId);
  };

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
    map.setVerticalFieldOfView(fieldOfView());
    map.on('move', () => setFieldOfView(map.getVerticalFieldOfView()));
    const lookAround = new LookAround(map, (available, active) => {
      setCanLookAround(available);
      setLookingAround(active);
    });
    toggleLookAround = () => lookAround.toggle();
    changeFieldOfView = (value) => lookAround.setFieldOfView(value);
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }));
    const attr = new maplibregl.AttributionControl({
      compact: false,
    });
    map.addControl(attr);

    const nightLayer = new NightLayer({
      opacity: opacity(),
      atmosphere: atmosphere(),
      exposure: exposure(),
      autoExposure: autoExposure(),
      updateInterval: 10000,
    });

    const captureView = () => {
      const view = lookAround.returnView;
      const center = view?.center ? maplibregl.LngLat.convert(view.center) : map.getCenter();
      const date = timeAnimating();
      return {
        center: [Number(center.lng.toFixed(8)), Number(center.lat.toFixed(8))],
        zoom: Number((view?.zoom ?? map.getZoom()).toFixed(3)),
        pitch: Number((view?.pitch ?? map.getPitch()).toFixed(3)),
        bearing: Number((view?.bearing ?? map.getBearing()).toFixed(3)),
        roll: Number((view?.roll ?? map.getRoll()).toFixed(3)),
        fov: Number(map.getVerticalFieldOfView().toFixed(3)),
        projection: globe() ? 'globe' : 'mercator',
        basemap: basemap(),
        terrain: terrain(),
        atmosphere: Number(atmosphere().toFixed(3)),
        exposure: Number(exposure().toFixed(3)),
        autoExposure: autoExposure(),
        opacity: Number(opacity().toFixed(3)),
        twilights: twilights(),
        timelineZoom: Number(zoom().toFixed(2)),
        date: date === null ? null : Number(date.toFixed(0)),
      };
    };
    copyView = async () => {
      const json = JSON.stringify(captureView());
      try {
        await navigator.clipboard.writeText(json);
      } catch {
        window.prompt('Copy view JSON', json);
      }
    };
    loadView = () => {
      const text = window.prompt('Load view JSON');
      if (text === null) return;
      try {
        const view = JSON.parse(text);
        const inRange = (value: unknown, min: number, max: number): value is number =>
          typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
        if (!view || typeof view !== 'object' || Array.isArray(view)) throw new Error('Invalid view JSON');
        const roll = view.roll === undefined ? 0 : view.roll;
        const fov = view.fov === undefined ? DEFAULT_FOV : view.fov;
        const timelineZoom = view.timelineZoom === undefined ? zoom() : view.timelineZoom;
        const basemapValue = view.basemap === undefined ? basemap() : view.basemap;
        const exposureValue = view.exposure === undefined ? 0 : view.exposure;
        const autoExposureValue = view.autoExposure === undefined ? false : view.autoExposure;
        const date = typeof view.date === 'string' ? Date.parse(view.date) / 1000 : view.date;
        if (
          !Array.isArray(view.center) ||
          view.center.length !== 2 ||
          !inRange(view.center[0], -Infinity, Infinity) ||
          !inRange(view.center[1], -90, 90) ||
          !inRange(view.zoom, map.getMinZoom(), map.getMaxZoom()) ||
          !inRange(view.pitch, 0, 85) ||
          !inRange(view.bearing, -Infinity, Infinity) ||
          !inRange(roll, -Infinity, Infinity) ||
          !inRange(fov, MIN_FOV, MAX_FOV) ||
          !['globe', 'mercator'].includes(view.projection) ||
          !['satellite', 'osm'].includes(basemapValue) ||
          typeof view.terrain !== 'boolean' ||
          !inRange(view.atmosphere, 0, 1) ||
          !inRange(exposureValue, -12, 12) ||
          typeof autoExposureValue !== 'boolean' ||
          !inRange(view.opacity, 0, 1) ||
          !Number.isInteger(view.twilights) ||
          !inRange(view.twilights, 0, 4) ||
          !inRange(timelineZoom, -35, 10) ||
          (date !== null && !inRange(date, -DATE_RANGE_SECONDS, DATE_RANGE_SECONDS))
        )
          throw new Error('Invalid view values');
        lookAround.stop();
        map.setVerticalFieldOfView(fov);
        setGlobe(view.projection === 'globe');
        setBasemap(basemapValue);
        setTerrain(view.terrain);
        setAtmosphere(view.atmosphere);
        setExposure(exposureValue);
        setAutoExposure(autoExposureValue);
        setOpacity(view.opacity);
        setTwilights(view.twilights);
        setZoom(timelineZoom);
        const selectedTime = date === null ? null : Decimal(date);
        setTime(selectedTime);
        setTimeAnimating(selectedTime);
        map.jumpTo({ center: view.center, zoom: view.zoom, pitch: view.pitch, bearing: view.bearing, roll });
      } catch (error) {
        window.alert(error instanceof Error ? error.message : 'Invalid view JSON');
      }
    };

    let urlTimer: ReturnType<typeof setTimeout>;
    const refreshURL = () => {
      clearTimeout(urlTimer);
      urlTimer = setTimeout(() => {
        const url = new URL(location.href);
        const view = captureView();
        const values = {
          center: view.center.map((value) => value.toFixed(8)).join(','),
          zoom: view.zoom.toFixed(3),
          pitch: view.pitch.toFixed(3),
          bearing: view.bearing.toFixed(3),
          roll: view.roll.toFixed(3),
          fov: view.fov.toFixed(3),
          projection: view.projection,
          basemap: view.basemap,
          terrain: view.terrain,
          atmosphere: view.atmosphere.toFixed(3),
          exposure: view.exposure.toFixed(3),
          autoExposure: view.autoExposure,
          opacity: view.opacity.toFixed(3),
          twilights: view.twilights.toFixed(0),
          timelineZoom: view.timelineZoom.toFixed(2),
        };
        for (const [key, value] of Object.entries(values)) url.searchParams.set(key, String(value));
        if (view.date === null) url.searchParams.delete('date');
        else url.searchParams.set('date', view.date.toFixed(0));
        if (url.href !== location.href) history.replaceState(history.state, '', url);
      }, 200);
    };
    map.on('move', refreshURL);
    effect(() => {
      globe();
      basemap();
      terrain();
      atmosphere();
      exposure();
      autoExposure();
      opacity();
      twilights();
      timeAnimating();
      zoom();
      refreshURL();
    });

    const [ready, setReady] = createSignal(false);
    map.on('load', () => {
      map.addLayer(nightLayer);
      map.addLayer(lookAround);
      setReady(true);
    });

    effect(() => {
      nightLayer.setTwilightSteps(twilights());
    });

    effect(() => {
      nightLayer.setOpacity(opacity());
    });
    effect(() => {
      nightLayer.setExposure(exposure());
    });
    effect(() => {
      nightLayer.setAutoExposure(autoExposure());
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
      <div class="settings">
        <button
          type="button"
          class="popup settings-toggle"
          popovertarget="settings-panel"
          aria-label="Settings"
          title="Settings"
        >
          <img src={SETTINGS_ICON} width="24" height="24" alt="" />
        </button>
        <button
          id="look-around"
          class="popup look-around-toggle"
          type="button"
          aria-label={() => (lookingAround() ? 'Return to map' : 'Look around')}
          title={() =>
            lookingAround() ? 'Return to map (Esc)' : 'Look around: available at zoom 12+ in fully Mercator rendering'
          }
          ref={(button) =>
            effect(() => {
              (button as HTMLButtonElement).disabled = !canLookAround() && !lookingAround();
            })
          }
          aria-pressed={lookingAround}
          onclick={() => toggleLookAround()}
        >
          <img src={LOOK_AROUND_ICON} width="24" height="24" alt="" />
        </button>
        <div class="popup field-of-view">
          <label for="field-of-view">FOV</label>
          <input
            id="field-of-view"
            type="range"
            min={Math.log10(MIN_FOV)}
            max={Math.log10(MAX_FOV)}
            step="0.0001"
            value={() => Math.log10(fieldOfView())}
            title="Vertical field of view: drag for fine adjustment; arrow keys for smaller steps"
            onpointerdown={startFovDrag}
            onpointermove={moveFovDrag}
            onpointerup={endFovDrag}
            onpointercancel={endFovDrag}
            onlostpointercapture={endFovDrag}
            oninput={(event: Event) => changeFieldOfView(10 ** Number((event.currentTarget as HTMLInputElement).value))}
          />
          <output for="field-of-view">{() => `${fieldOfView().toFixed(fieldOfView() < 10 ? 2 : 1)}°`}</output>
          <button
            type="button"
            class="fov-reset"
            aria-label="Reset field of view"
            title="Reset field of view"
            onclick={() => changeFieldOfView(DEFAULT_FOV)}
          >
            <img src={RESTORE_ICON} width="18" height="18" alt="" />
          </button>
        </div>
        <div id="settings-panel" class="popup display-controls" popover="auto" aria-label="Settings">
          <p class="look-around-help">
            {() =>
              lookingAround()
                ? 'Drag to look; Shift + drag to move horizontally. Wheel to move forward/back; Shift + wheel to change altitude. Esc to resume the map facing this direction (up to 85°). URL/COPY keep the entry view.'
                : 'Available at zoom 12+ in fully Mercator rendering.'
            }
          </p>
          <div>
            <button id="copy-view" type="button" onclick={() => copyView()}>
              COPY
            </button>
            <button id="load-view" type="button" onclick={() => loadView()}>
              LOAD
            </button>
          </div>
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
            <span id="auto-exposure-label">Auto exposure</span>
            <button
              id="btn-auto-exposure"
              type="button"
              class="toggle-switch"
              role="switch"
              aria-labelledby="auto-exposure-label"
              aria-checked={autoExposure}
              onclick={() => setAutoExposure((v) => !v)}
            >
              <span class="toggle-track" aria-hidden="true" />
            </button>
          </div>
          <div>
            <label for="exposure">{() => (autoExposure() ? 'Exposure comp.' : 'Exposure')}</label>
            <input
              id="exposure"
              type="range"
              min="-12"
              max="12"
              step="0.1"
              value={exposure}
              oninput={(e: Event) => setExposure(Number((e.currentTarget as HTMLInputElement).value))}
            />
            <output for="exposure">{() => `${exposure() > 0 ? '+' : ''}${exposure().toFixed(1)} stops`}</output>
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
                effect(() => {
                  const value = String(twilights());
                  queueMicrotask(() => {
                    (select as HTMLSelectElement).value = value;
                  });
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
                effect(() => {
                  const value = basemap();
                  queueMicrotask(() => {
                    (select as HTMLSelectElement).value = value;
                  });
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
              disabled={lookingAround}
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
              disabled={lookingAround}
              onclick={() => setTerrain((v) => !v)}
            >
              <span class="toggle-track" aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

const app = document.getElementById('app');
if (app) {
  render(app, <App />);
}
