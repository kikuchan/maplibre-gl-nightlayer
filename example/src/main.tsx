import { createMemo, createSignal, effect, onMount, render } from '@luna_ui/luna';
import * as maplibregl from 'maplibre-gl';
import maplibreWorkerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url';
import 'maplibre-gl/dist/maplibre-gl.css';
import './main.css';
import {
  NightLayer,
  type Atmosphere,
  type AtmosphereInput,
  type Development,
  type DevelopmentInput,
  type Twilight,
  type TwilightInput,
  type TwilightSteps,
} from '../../src/main';
import atmosphereLutUrl from '../../src/atmosphere-lut.zip?url';
import { focalLength, SENSOR_PITCH_MM, MAX_FOV, MIN_FOV } from './camera-settings';
import { locationControl } from './location-control';
import { LookAround, cameraViewSettings } from './look-around';
import { Timescope } from '@timescope/luna';
import { Decimal, Calendar } from 'timescope';

maplibregl.setWorkerUrl(maplibreWorkerUrl);

const DATE_RANGE_SECONDS = 8_640_000_000_000;
const DEFAULT_GRADE: Development = {
  process: 'aces',
  exposure: 0,
  autoExposure: false,
  compensation: 0,
  gamma: 1,
};

// Expand API shorthand into values for the example's individual UI controls.
function atmosphereControls(input: AtmosphereInput): Atmosphere {
  if (typeof input === 'boolean' || typeof input === 'number') {
    const strength = Number(input);
    return { sky: strength, fog: strength, stars: strength > 0 };
  }
  return { sky: Number(input.sky ?? true), fog: Number(input.fog ?? true), stars: input.stars ?? true };
}

function developmentControls(input: DevelopmentInput): Development | false {
  if (input === false) return false;
  return { ...DEFAULT_GRADE, ...(typeof input === 'string' ? { process: input } : input) };
}

function twilightControls(input?: TwilightInput): Twilight | null {
  if (input == null) return null;
  const value = typeof input === 'number' ? { steps: input } : input;
  const color = value.color ?? [0, 0, 0, 0.95];
  return {
    steps: value.steps,
    color: [color[0], color[1], color[2], color[3] ?? 0.95],
    attenuation: value.attenuation ?? 0.5,
  };
}

const SETTINGS_ICON = `data:image/svg+xml,${encodeURIComponent(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#f1f5f9" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="m9.5 3-.6 2.1-1.6.9-2.1-.5-2.5 4.3 1.5 1.6v1.8l-1.5 1.6 2.5 4.3 2.1-.5 1.6.9.6 2.1h5l.6-2.1 1.6-.9 2.1.5 2.5-4.3-1.5-1.6v-1.8l1.5-1.6-2.5-4.3-2.1.5-1.6-.9L14.5 3Z"/>
    <circle cx="12" cy="12" r="3"/>
  </svg>
`)}`;
const CAMERA_ICON = `data:image/svg+xml,${encodeURIComponent(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#f1f5f9" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 7h4l2-3h6l2 3h4v13H3Z"/>
    <circle cx="12" cy="13" r="4"/>
  </svg>
`)}`;
const RESET_ICON = `data:image/svg+xml,${encodeURIComponent(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 24" fill="none" stroke="#fff" stroke-width="1.25" stroke-linecap="square" stroke-linejoin="bevel">
    <path d="M4 8h6l2 2v6l-2 2H4l-2-2v-3M6 5 3 8l3 3"/>
  </svg>
`)}`;
const ALL_SKY_ICON = `data:image/svg+xml,${encodeURIComponent(`
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none" stroke="#f1f5f9" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M22 12C22-2 10-2 10 12s12 14 12 6"/>
    <path d="M16 18C-2 18-2 6 16 6s18 12 0 12m3-3-3 3 3 3"/>
    <path d="m18.5 8.5 3.5 3.5 3.5-3.5"/>
    <text x="16" y="31" text-anchor="middle" fill="#f1f5f9" stroke="none" font-family="Arial, sans-serif" font-size="7" font-weight="600">360°</text>
  </svg>
`)}`;
type Basemap = 'satellite' | 'osm';
type CameraDevelopment = Development['process'];
const CAMERA_DEVELOPMENTS = [
  { value: 'agx', label: 'AgX' },
  { value: 'aces', label: 'ACES' },
  { value: 'filmic', label: 'Filmic' },
  { value: 'linear', label: 'Linear' },
] as const;
const isCameraDevelopment = (value: unknown): value is CameraDevelopment =>
  CAMERA_DEVELOPMENTS.some((option) => option.value === value);
const LEGACY_LIGHTING_PARAMS = [
  'tileGamma',
  'twilights',
  'localTone',
  'fNumber',
  'shutterSeconds',
  'iso',
  'focusDistanceMeters',
  'depthOfField',
  'starBrightness',
  'nightGroundBrightness',
  'starDarkLimit',
  'starBrightLimit',
  'nightDarkLimit',
  'nightBrightLimit',
  'finiteMieScattering',
  'terrainMieShadows',
  'terrainShadowDebug',
  'terrainShadowMapDebug',
  'directTerrainShadows',
  'terrainShadowBias',
  'terrainShadowScale',
  'terrainShadowDebugDistance',
  'lut',
  'filmicModel',
  'filmicBlack',
  'filmicWhite',
  'filmicGrey',
  'filmicContrast',
  'filmicPoints',
  'logBlack',
  'logWhite',
  'linearCurve',
  'linearExposure',
  'cameraDevelopment',
  'displayGamma',
  'autoExposure',
  'autoExposureCompensation',
  'exposure',
  'mode',
  'stars',
];

function CameraReading({
  label,
  value,
  options,
  format,
  change,
  visible = () => true,
}: {
  label: string;
  value: () => number;
  options: number[];
  format: (value: number) => string;
  change: (value: number) => void;
  visible?: () => boolean;
}) {
  return (
    <label
      class="camera-reading"
      title={label}
      ref={(element) =>
        effect(() => {
          element.hidden = !visible();
        })
      }
    >
      <select
        aria-label={label}
        ref={(element) =>
          effect(() => {
            const current = value();
            const preset = options.find(
              (option) =>
                option === current ||
                (Number.isFinite(option) &&
                  Number.isFinite(current) &&
                  Math.abs(option - current) <= 1e-8 * Math.max(1, Math.abs(current))),
            );
            (element as HTMLSelectElement).value = preset === undefined ? 'custom' : String(preset);
            element.style.width = `${format(current).length + 1}ch`;
          })
        }
        onchange={(event: Event) => {
          const select = event.currentTarget as HTMLSelectElement;
          if (select.value !== 'custom') change(Number(select.value));
        }}
      >
        {options.map((option) => (
          <option value={String(option)}>{format(option)}</option>
        ))}
        <option value="custom" hidden disabled>
          {() => format(value())}
        </option>
      </select>
    </label>
  );
}

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
  const errorMessages = {
    page: {
      title: 'ページ内エラー / Page error',
      description: 'ページ内でエラーが報告されました。地図の描画エラーとは限りません。',
    },
    map: {
      title: '地図エラー / Map error',
      description: '地図の処理中にエラーが発生しました。',
    },
    layer: {
      title: 'NightLayerエラー / NightLayer error',
      description: 'NightLayerの初期化または描画中にエラーが発生しました。',
    },
  };
  const [runtimeError, setRuntimeError] = createSignal('');
  const [runtimeErrorSource, setRuntimeErrorSource] = createSignal<'page' | 'map' | 'layer'>('page');
  const [runtimeDiagnostics, setRuntimeDiagnostics] = createSignal(
    `${navigator.userAgent}\nFloat16Array: ${typeof Float16Array !== 'undefined' ? 'supported' : 'missing'}`,
  );
  const reportError = (error: unknown, context?: string, source: 'page' | 'map' | 'layer' = 'page') => {
    const value = error && typeof error === 'object' ? (error as { message?: unknown; stack?: unknown }) : undefined;
    const message = typeof value?.message === 'string' ? value.message : String(error);
    const details = [message, context, typeof value?.stack === 'string' ? value.stack : undefined]
      .filter(Boolean)
      .join('\n');
    const previous = runtimeError();
    // Prefer errors caught at their source; an opaque page error is not proof of rendering failure.
    const priority = { page: 0, map: 1, layer: 2 };
    if (previous && !previous.startsWith('Script error.') && priority[source] <= priority[runtimeErrorSource()]) return;
    setRuntimeErrorSource(source);
    setRuntimeError(details);
  };
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
  const [coordinateText, setCoordinateText] = createSignal(initialCenter.join(', '));
  const [time, setTime] = createSignal<Decimal | null>(initialTime);
  const [timeAnimating, setTimeAnimating] = createSignal<Decimal | null>(initialTime);
  const [zoom, setZoom] = createSignal(numberParam('timelineZoom', -8, -35, 10));

  let initialTwilight: Twilight | null = null;
  try {
    initialTwilight = twilightControls(JSON.parse(query.get('twilight') ?? 'null'));
  } catch {}
  const [twilight, setTwilight] = createSignal<Twilight | null>(initialTwilight);
  const updateTwilight = (changes: Partial<Twilight>) =>
    setTwilight((previous) => (previous ? { ...previous, ...changes } : null));
  const twilightColorHex = () =>
    '#' +
    (twilight()?.color ?? [0, 0, 0])
      .slice(0, 3)
      .map((v) =>
        Math.round(v * 255)
          .toString(16)
          .padStart(2, '0'),
      )
      .join('');
  const [opacity, setOpacity] = createSignal(numberParam('opacity', 1, 0, 1));
  const [globe, setGlobe] = createSignal(query.get('projection') !== 'mercator');
  const [basemap, setBasemap] = createSignal<Basemap>(query.get('basemap') === 'osm' ? 'osm' : 'satellite');
  let initialAtmosphere = atmosphereControls(true);
  let initialDevelopment: Development | false = false;
  try {
    initialAtmosphere = atmosphereControls(JSON.parse(query.get('atmosphere') ?? 'true'));
  } catch {}
  try {
    initialDevelopment = developmentControls(JSON.parse(query.get('development') ?? 'false'));
  } catch {}
  const initialGrade = initialDevelopment || DEFAULT_GRADE;
  const [sky, setSky] = createSignal(initialAtmosphere.sky);
  const [fog, setFog] = createSignal(initialAtmosphere.fog);
  const [displayGamma, setDisplayGamma] = createSignal(initialGrade.gamma);
  const [exposure, setExposure] = createSignal(initialGrade.exposure);
  const [autoExposure, setAutoExposure] = createSignal(initialGrade.autoExposure);
  const [autoExposureCompensation, setAutoExposureCompensation] = createSignal(initialGrade.compensation);
  const [cameraDevelopment, setCameraDevelopment] = createSignal<CameraDevelopment>(initialGrade.process);
  const [stars, setStars] = createSignal(initialAtmosphere.stars);
  const [terrain, setTerrain] = createSignal(query.get('terrain') === 'true');
  const [cameraMode, setCameraMode] = createSignal(initialDevelopment !== false);
  const [allSky, setAllSky] = createSignal(false);
  const [canAllSky, setCanAllSky] = createSignal(false);
  const [logicalHeight, setLogicalHeight] = createSignal(Math.max(1, window.innerHeight));
  const sensorHeight = () => logicalHeight() * SENSOR_PITCH_MM;
  const fovForFocal = (value: number) =>
    Math.max(MIN_FOV, Math.min(MAX_FOV, (2 * Math.atan(sensorHeight() / (2 * value)) * 180) / Math.PI));
  const [fieldOfView, setFieldOfView] = createSignal(fovForFocal(36));
  const [capturing, setCapturing] = createSignal(false);
  const atmosphereView = () => ({ sky: sky(), fog: fog(), stars: stars() });
  const developmentView = (): Development | false =>
    cameraMode()
      ? {
          process: cameraDevelopment(),
          exposure: exposure(),
          autoExposure: autoExposure(),
          compensation: autoExposureCompensation(),
          gamma: displayGamma(),
        }
      : false;
  let captureEXR = async () => {};

  const [animating, setAnimating] = createSignal(false);
  const [editing, setEditing] = createSignal(false);
  const [now, setNow] = createSignal(Date.now());
  let copyView = async () => {};
  let loadView = () => {};
  let changeCoordinates = (_input: HTMLInputElement) => {};
  let toggleAllSky = () => {};

  onMount(() => {
    window.addEventListener('error', (event) =>
      reportError(
        event.error ?? event.message,
        event.filename ? `${event.filename}:${event.lineno}:${event.colno}` : undefined,
      ),
    );
    window.addEventListener('unhandledrejection', (event) => reportError(event.reason, 'Unhandled promise rejection'));
    let map: maplibregl.Map;
    try {
      map = new maplibregl.Map({
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
            white: {
              type: 'raster',
              tiles: ['white.png'],
              tileSize: 256,
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
    } catch (error) {
      reportError(error, 'Map initialization', 'map');
      return;
    }
    map.on('error', (event) => {
      // Missing individual basemap/DEM tiles are recoverable, unlike layer initialization failures.
      if ('sourceId' in event) return;
      reportError(event.error, 'MapLibre', 'map');
    });
    const gl = map.getCanvas().getContext('webgl2');
    const extensions = gl?.getSupportedExtensions() ?? [];
    const floatBuffers = gl?.getExtension('EXT_color_buffer_float');
    const samples = (format: number): string => {
      const counts = gl?.getInternalformatParameter(gl.RENDERBUFFER, format, gl.SAMPLES) as Int32Array | null;
      return counts?.length ? Array.from(counts).join(', ') : 'none';
    };
    setRuntimeDiagnostics(
      [
        runtimeDiagnostics(),
        `WebGL 2: ${gl ? 'supported' : 'missing'}`,
        ...['OES_texture_float_linear', 'EXT_color_buffer_float', 'EXT_float_blend'].map(
          (name) => `${name}: ${extensions.includes(name) ? 'supported' : 'missing'}`,
        ),
        ...(gl
          ? [
              `MAX_TEXTURE_SIZE: ${gl.getParameter(gl.MAX_TEXTURE_SIZE)}`,
              `MAX_ARRAY_TEXTURE_LAYERS: ${gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS)}`,
              `MAX_TEXTURE_IMAGE_UNITS: ${gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)}`,
              `MAX_RENDERBUFFER_SIZE: ${gl.getParameter(gl.MAX_RENDERBUFFER_SIZE)}`,
              `Canvas: ${gl.drawingBufferWidth}×${gl.drawingBufferHeight} device pixels; ${map.getCanvas().clientWidth}×${map.getCanvas().clientHeight} CSS pixels`,
            ]
          : []),
        ...(gl && floatBuffers
          ? [
              `RGBA32F MSAA samples: ${samples(gl.RGBA32F)}`,
              `DEPTH24_STENCIL8 MSAA samples: ${samples(gl.DEPTH24_STENCIL8)}`,
            ]
          : []),
      ].join('\n'),
    );
    map.on('webglcontextlost', () => reportError(new Error('WebGL context lost'), 'Map canvas', 'map'));
    map.setVerticalFieldOfView(fieldOfView());
    const lookAround = new LookAround(map, (available, active) => {
      const changed = active !== allSky();
      setCanAllSky(available);
      setAllSky(active);
      if (changed) {
        const fov = fovForFocal(36);
        lookAround.resizeFieldOfView(fov, fov);
        setFieldOfView(fov);
      }
    });
    toggleAllSky = () => lookAround.toggle();
    map.on('move', () => setFieldOfView(map.getVerticalFieldOfView()));
    window.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      if (allSky()) lookAround.stop();
      else setCameraMode(false);
    });
    const refreshCoordinates = () => {
      if (document.activeElement?.id === 'coordinates') return;
      const center = map.getCenter();
      const observer = lookAround.cameraView?.observer;
      setCoordinateText(`${(observer?.[0] ?? center.lng).toFixed(8)}, ${(observer?.[1] ?? center.lat).toFixed(8)}`);
    };
    map.on('move', refreshCoordinates);
    changeCoordinates = (input) => {
      const parts = input.value.trim().split(/[,\s]+/);
      const [longitude, latitude] = parts.map(Number);
      const camera = lookAround.cameraView;
      const latitudeLimit = camera ? 85.051129 : 90;
      if (
        parts.length !== 2 ||
        !Number.isFinite(longitude) ||
        !Number.isFinite(latitude) ||
        Math.abs(longitude) > 180 ||
        Math.abs(latitude) > latitudeLimit
      ) {
        input.setCustomValidity(`Enter longitude, latitude in degrees (longitude ±180, latitude ±${latitudeLimit}).`);
        input.reportValidity();
        return;
      }
      input.setCustomValidity('');
      if (camera) lookAround.restoreCamera({ ...camera, observer: [longitude, latitude, camera.observer[2]] });
      else map.jumpTo({ center: [longitude, latitude] });
      setCoordinateText(`${longitude}, ${latitude}`);
    };
    const resizeSensor = () => {
      const height = Math.max(1, map.getCanvas().clientHeight);
      if (height === logicalHeight()) return;
      const lens = allSky() ? focalLength(fieldOfView(), sensorHeight()) : 36;
      setLogicalHeight(height);
      const fov = fovForFocal(lens);
      setFieldOfView(fov);
      lookAround.resizeFieldOfView(fov, fovForFocal(36));
    };
    map.on('resize', resizeSensor);
    resizeSensor();
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }));
    map.addControl(locationControl(), 'top-right');
    const attr = new maplibregl.AttributionControl({
      compact: false,
      customAttribution:
        'Stars: <a href="https://github.com/astronexus/HYG-Database">HYG / D. Nash</a> (<a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>), <a href="https://svs.gsfc.nasa.gov/4851/">NASA/GSFC/SVS</a>, <a href="https://www.cosmos.esa.int/web/gaia/dpac">ESA/Gaia/DPAC</a>',
    });
    map.addControl(attr);

    const nightLayer = new NightLayer({
      opacity: opacity(),
      atmosphere: atmosphereView(),
      development: developmentView(),
      lut: atmosphereLutUrl,
      updateInterval: 10000,
    });
    // Capture every rendering phase before WebKit can redact a cross-script exception.
    // In particular, MSAA allocation and native surface shaders happen in prerender.
    const captureLayerError =
      <Args extends unknown[]>(phase: string, callback: (...args: Args) => void) =>
      (...args: Args): void => {
        try {
          callback(...args);
        } catch (error) {
          reportError(error, phase, 'layer');
          throw error;
        }
      };
    nightLayer.onAdd = captureLayerError('NightLayer.onAdd', nightLayer.onAdd.bind(nightLayer));
    nightLayer.prerender = captureLayerError('NightLayer.prerender', nightLayer.prerender.bind(nightLayer));
    nightLayer.render = captureLayerError('NightLayer.render', nightLayer.render.bind(nightLayer));
    map.on('render', () => {
      const development = nightLayer.getDevelopment();
      if (cameraMode() && development && development.process === cameraDevelopment()) setExposure(development.exposure);
    });

    const captureView = () => {
      const entry = lookAround.returnView;
      const center = entry?.center ? maplibregl.LngLat.convert(entry.center) : map.getCenter();
      const date = timeAnimating();
      return {
        development: developmentView(),
        allSky: lookAround.cameraView,
        center: [Number(center.lng.toFixed(8)), Number(center.lat.toFixed(8))],
        elevation: entry?.elevation ?? map.getCenterElevation(),
        zoom: Number((entry?.zoom ?? map.getZoom()).toFixed(3)),
        pitch: Number((entry?.pitch ?? map.getPitch()).toFixed(3)),
        bearing: Number((entry?.bearing ?? map.getBearing()).toFixed(3)),
        roll: Number((entry?.roll ?? map.getRoll()).toFixed(3)),
        fov: Number(fieldOfView().toFixed(3)),
        focalLengthMm: Number(focalLength(fieldOfView(), sensorHeight()).toFixed(6)),
        projection: globe() ? 'globe' : 'mercator',
        basemap: basemap(),
        terrain: terrain(),
        atmosphere: atmosphereView(),
        opacity: Number(opacity().toFixed(3)),
        twilight: twilight(),
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
    captureEXR = async () => {
      if (capturing()) return;
      setCapturing(true);
      try {
        const blob = await nightLayer.captureEXR();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `nightlayer-${new Date().toISOString().replaceAll(':', '-')}.exr`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      } catch (error) {
        window.alert(error instanceof Error ? error.message : 'EXR capture failed');
      } finally {
        setCapturing(false);
      }
    };
    const applyView = (text: string) => {
      try {
        const view = JSON.parse(text);
        const inRange = (value: unknown, min: number, max: number): value is number =>
          typeof value === 'number' && Number.isFinite(value) && value >= min && value <= max;
        if (!view || typeof view !== 'object' || Array.isArray(view)) throw new Error('Invalid view JSON');
        const roll = view.roll === undefined ? 0 : view.roll;
        const timelineZoom = view.timelineZoom === undefined ? zoom() : view.timelineZoom;
        const basemapValue = view.basemap === undefined ? basemap() : view.basemap;
        const development = developmentControls(view.development ?? false);
        const grade = development || DEFAULT_GRADE;
        const atmosphere = atmosphereControls(view.atmosphere ?? false);
        const twilightValue = twilightControls(view.twilight);
        const skyCamera = view.allSky == null ? undefined : cameraViewSettings(view.allSky);
        const elevation = view.elevation;
        const date = typeof view.date === 'string' ? Date.parse(view.date) / 1000 : view.date;
        if (
          !Array.isArray(view.center) ||
          (skyCamera !== undefined && view.zoom < 12) ||
          (elevation !== undefined && !inRange(elevation, -Infinity, Infinity)) ||
          view.center.length !== 2 ||
          !inRange(view.center[0], -Infinity, Infinity) ||
          !inRange(view.center[1], -90, 90) ||
          !inRange(view.zoom, map.getMinZoom(), map.getMaxZoom()) ||
          !inRange(view.pitch, 0, 85) ||
          !inRange(view.bearing, -Infinity, Infinity) ||
          !inRange(roll, -Infinity, Infinity) ||
          !['globe', 'mercator'].includes(view.projection) ||
          !['satellite', 'osm'].includes(basemapValue) ||
          typeof view.terrain !== 'boolean' ||
          !inRange(view.opacity, 0, 1) ||
          !inRange(timelineZoom, -35, 10) ||
          (date !== null && !inRange(date, -DATE_RANGE_SECONDS, DATE_RANGE_SECONDS))
        )
          throw new Error('Invalid view values');
        lookAround.stop();
        lookAround.setFieldOfView(fovForFocal(36));
        setFieldOfView(fovForFocal(36));
        setCameraMode(development !== false);
        setGlobe(view.projection === 'globe');
        setBasemap(basemapValue);
        setTerrain(view.terrain);
        setSky(atmosphere.sky);
        setFog(atmosphere.fog);
        setStars(atmosphere.stars);
        setDisplayGamma(grade.gamma);
        setExposure(grade.exposure);
        setAutoExposure(grade.autoExposure);
        setAutoExposureCompensation(grade.compensation);
        setCameraDevelopment(grade.process);
        setOpacity(view.opacity);
        setTwilight(twilightValue);
        setZoom(timelineZoom);
        const selectedTime = date === null ? null : Decimal(date);
        setTime(selectedTime);
        setTimeAnimating(selectedTime);
        map.setProjection({ type: view.projection });
        map.jumpTo({ center: view.center, zoom: view.zoom, pitch: view.pitch, bearing: view.bearing, roll, elevation });
        if (skyCamera) lookAround.restoreCamera(skyCamera);
      } catch (error) {
        window.alert(error instanceof Error ? error.message : 'Invalid view JSON');
      }
    };
    loadView = () => {
      const text = window.prompt('Load view JSON');
      if (text !== null) applyView(text);
    };

    let urlTimer: ReturnType<typeof setTimeout>;
    const refreshURL = () => {
      clearTimeout(urlTimer);
      urlTimer = setTimeout(() => {
        const url = new URL(location.href);
        const view = captureView();
        const values = {
          development: JSON.stringify(view.development),
          center: view.center.map((value) => value.toFixed(8)).join(','),
          zoom: view.zoom.toFixed(3),
          pitch: view.pitch.toFixed(3),
          bearing: view.bearing.toFixed(3),
          roll: view.roll.toFixed(3),
          fov: view.fov.toFixed(3),
          focalLengthMm: view.focalLengthMm,
          projection: view.projection,
          basemap: view.basemap,
          terrain: view.terrain,
          atmosphere: JSON.stringify(view.atmosphere),
          opacity: view.opacity.toFixed(3),
          twilight: JSON.stringify(view.twilight),
          timelineZoom: view.timelineZoom.toFixed(2),
        };
        for (const [key, value] of Object.entries(values)) url.searchParams.set(key, String(value));
        if (view.allSky) url.searchParams.set('allSky', JSON.stringify(view.allSky));
        else url.searchParams.delete('allSky');
        for (const key of LEGACY_LIGHTING_PARAMS) url.searchParams.delete(key);
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
      atmosphereView();
      developmentView();
      opacity();
      twilight();
      timeAnimating();
      zoom();
      allSky();
      fieldOfView();
      refreshURL();
    });

    const [ready, setReady] = createSignal(false);
    map.on('load', () => {
      try {
        map.addLayer(nightLayer);
        map.addLayer(lookAround);
        if (query.has('allSky')) {
          try {
            const camera = cameraViewSettings(JSON.parse(query.get('allSky')!));
            lookAround.restoreCamera(camera);
          } catch {
            // Invalid observer state must not prevent the normal map from loading.
          }
        }
        setReady(true);
      } catch (error) {
        reportError(error, 'Map load', 'map');
      }
    });

    effect(() => {
      nightLayer.setTwilight(twilight());
    });

    effect(() => {
      nightLayer.setOpacity(opacity());
    });
    effect(() => {
      nightLayer.setDevelopment(developmentView());
    });

    effect(() => {
      const t = timeAnimating();
      nightLayer.setDate(t === null ? null : new Date(t.mul(1000).number()));
    });

    effect(() => {
      const type = allSky() ? 'mercator' : globe() ? 'globe' : 'mercator';
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
      nightLayer.setAtmosphere(atmosphereView());
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
      <div id="runtime-error" role="alert" style={() => `display:${runtimeError() ? 'block' : 'none'};`}>
        <strong>{() => errorMessages[runtimeErrorSource()].title}</strong>
        <pre>{runtimeError}</pre>
        <p>
          {() =>
            runtimeError().startsWith('Script error.')
              ? 'ブラウザがエラーの詳細を非公開にしています。発生元は未特定で、描画の停止を示すものではありません。'
              : errorMessages[runtimeErrorSource()].description
          }
        </p>
        <pre>{runtimeDiagnostics}</pre>
        <button type="button" onclick={() => setRuntimeError('')}>
          閉じる / Close
        </button>
      </div>
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
      <div
        class="camera-hud"
        ref={(element) =>
          effect(() => {
            (element as HTMLElement).hidden = !cameraMode();
          })
        }
        role="group"
        aria-label="Camera development controls"
      >
        <div class="camera-exposure-controls">
          <label class="camera-exposure" for="camera-exposure">
            <input
              id="camera-exposure"
              type="range"
              min="-12"
              max="32"
              step="0.1"
              value={exposure}
              aria-label="Exposure (EV)"
              oninput={(event: Event) => {
                setAutoExposure(false);
                setExposure(Number((event.currentTarget as HTMLInputElement).value));
              }}
            />
            <output for="camera-exposure">{() => `EV${exposure() >= 0 ? '+' : ''}${exposure().toFixed(1)}`}</output>
          </label>
          <button
            id="btn-auto-exposure"
            type="button"
            class="camera-ae"
            aria-label="Auto exposure"
            aria-pressed={autoExposure}
            onclick={() => setAutoExposure((value) => !value)}
          >
            AE {() => (autoExposure() ? 'ON' : 'OFF')}
          </button>
          <button
            type="button"
            class="camera-reset"
            aria-label="Reset exposure"
            title="Reset exposure to EV 0 and turn AE off"
            onclick={() => {
              setAutoExposure(false);
              setAutoExposureCompensation(0);
              setExposure(0);
            }}
          >
            <img src={RESET_ICON} width="14" height="24" alt="" />
          </button>
        </div>
        <label class="camera-reading">
          <select
            aria-label="Camera development"
            title="Develop: ACES, AgX, RGB ratio, Filmic Blender Base Contrast (approximate), or RGB Curve"
            ref={(element) =>
              effect(() => {
                (element as HTMLSelectElement).value = cameraDevelopment();
                element.style.width = `${CAMERA_DEVELOPMENTS.find((option) => option.value === cameraDevelopment())!.label.length + 1}ch`;
              })
            }
            onchange={(event: Event) => {
              const select = event.currentTarget as HTMLSelectElement;
              if (isCameraDevelopment(select.value)) setCameraDevelopment(select.value);
            }}
          >
            {CAMERA_DEVELOPMENTS.map((option) => (
              <option value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>
        <div class="camera-exposure-controls">
          <label class="camera-exposure camera-gamma" for="display-gamma">
            <span>Gamma</span>
            <input
              id="display-gamma"
              type="range"
              min="0.1"
              max="4"
              step="0.01"
              value={displayGamma}
              aria-label="Display gamma"
              title="After development: below 1 darkens midtones; above 1 brightens. Does not affect AE or EXR."
              oninput={(event: Event) => setDisplayGamma(Number((event.currentTarget as HTMLInputElement).value))}
            />
            <output for="display-gamma">{() => displayGamma().toFixed(2)}</output>
          </label>
          <button
            type="button"
            class="camera-reset"
            aria-label="Reset display gamma"
            title="Reset display gamma to 1"
            onclick={() => setDisplayGamma(1)}
          >
            <img src={RESET_ICON} width="14" height="24" alt="" />
          </button>
        </div>
        <CameraReading
          label="AE bias"
          visible={autoExposure}
          value={autoExposureCompensation}
          options={Array.from({ length: 49 }, (_, i) => -12 + i * 0.5)}
          format={(value) => `EC${value >= 0 ? '+' : ''}${Number(value.toFixed(1))}`}
          change={setAutoExposureCompensation}
        />
        <button
          type="button"
          ref={(button) =>
            effect(() => {
              (button as HTMLButtonElement).disabled = capturing();
            })
          }
          aria-label="Save linear RGB as OpenEXR"
          title="Save float32 linear RGB before preview exposure and tone mapping"
          onclick={() => captureEXR()}
        >
          {() => (capturing() ? 'Saving…' : 'EXR ↓')}
        </button>
      </div>
      <div class="settings">
        <button
          type="button"
          popovertarget="settings-panel"
          aria-label="Settings"
          title="Settings"
        >
          <img src={SETTINGS_ICON} width="24" height="24" alt="" />
        </button>
        <button
          id="look-around"
          type="button"
          aria-label={() => (cameraMode() ? 'Exit camera mode' : 'Camera mode')}
          title={() =>
            cameraMode()
              ? 'Exit camera mode (Esc)'
              : 'Camera mode: control exposure and development with normal map navigation'
          }
          aria-pressed={cameraMode}
          onclick={() => setCameraMode((value) => !value)}
        >
          <img src={CAMERA_ICON} width="24" height="24" alt="" />
        </button>
        <button
          id="all-sky"
          type="button"
          aria-label={() => (allSky() ? 'Exit all-sky view' : 'All-sky view')}
          title={() =>
            allSky()
              ? 'Exit all-sky view (Esc)'
              : 'All-sky view: look around from a fixed observer at zoom 12+ in fully Mercator rendering'
          }
          ref={(button) =>
            effect(() => {
              (button as HTMLButtonElement).disabled = !canAllSky() && !allSky();
            })
          }
          aria-pressed={allSky}
          onclick={() => toggleAllSky()}
        >
          <img src={ALL_SKY_ICON} width="28" height="28" alt="" />
        </button>
        <div id="settings-panel" class="popup display-controls" popover="auto" aria-label="Settings">
          <div>
            <button id="copy-view" type="button" onclick={() => copyView()}>
              COPY
            </button>
            <button id="load-view" type="button" onclick={() => loadView()}>
              LOAD
            </button>
          </div>
          <form
            class="coordinate-entry"
            onsubmit={(event: Event) => {
              event.preventDefault();
              changeCoordinates(
                (event.currentTarget as HTMLFormElement).elements.namedItem('coordinates') as HTMLInputElement,
              );
            }}
          >
            <label for="coordinates">Longitude, latitude (°)</label>
            <input
              id="coordinates"
              name="coordinates"
              type="text"
              value={coordinateText}
              placeholder="135, 35"
              required
              oninput={(event: Event) => {
                const input = event.currentTarget as HTMLInputElement;
                input.setCustomValidity('');
                setCoordinateText(input.value);
              }}
            />
            <button type="submit">Go</button>
          </form>
          <div>
            <label for="atmosphere-blend">Sky / Sun / Moon</label>
            <input
              id="atmosphere-blend"
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={sky}
              oninput={(e: Event) => setSky(Number((e.currentTarget as HTMLInputElement).value))}
            />
            <output for="atmosphere-blend">{() => sky().toFixed(2)}</output>
          </div>
          <div>
            <label for="fog-blend">Fog</label>
            <input
              id="fog-blend"
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={fog}
              oninput={(event: Event) => setFog(Number((event.currentTarget as HTMLInputElement).value))}
            />
            <output for="fog-blend">{() => fog().toFixed(2)}</output>
          </div>
          <div>
            <span id="stars-label">Stars</span>
            <button
              id="btn-stars"
              type="button"
              class="toggle-switch"
              role="switch"
              aria-labelledby="stars-label"
              aria-checked={stars}
              onclick={() => setStars((value) => !value)}
            >
              <span class="toggle-track" aria-hidden="true" />
            </button>
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
            <label for="twilight-steps">Twilight</label>
            <select
              id="twilight-steps"
              value={() => twilight()?.steps ?? 0}
              disabled={cameraMode}
              ref={(select) =>
                effect(() => {
                  const value = String(twilight()?.steps ?? 0);
                  queueMicrotask(() => {
                    (select as HTMLSelectElement).value = value;
                  });
                })
              }
              onchange={(e: Event) => {
                const steps = Number((e.currentTarget as HTMLSelectElement).value) as TwilightSteps | 0;
                setTwilight(steps === 0 ? null : twilightControls({ ...twilight(), steps }));
              }}
            >
              <option value="0">Continuous</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
            </select>
          </div>
          <div>
            <label for="twilight-color">Night color</label>
            <input
              id="twilight-color"
              type="color"
              value={twilightColorHex}
              disabled={() => cameraMode() || twilight() === null}
              oninput={(e: Event) => {
                const hex = (e.currentTarget as HTMLInputElement).value;
                const rgb: [number, number, number] = [
                  parseInt(hex.slice(1, 3), 16) / 255,
                  parseInt(hex.slice(3, 5), 16) / 255,
                  parseInt(hex.slice(5, 7), 16) / 255,
                ];
                updateTwilight({ color: [...rgb, twilight()?.color[3] ?? 0.95] });
              }}
            />
          </div>
          <div>
            <label for="twilight-opacity">Night opacity</label>
            <input
              id="twilight-opacity"
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={() => twilight()?.color[3] ?? 0.95}
              disabled={() => cameraMode() || twilight() === null}
              oninput={(e: Event) => {
                const color = twilight()?.color ?? [0, 0, 0, 0.95];
                updateTwilight({
                  color: [color[0], color[1], color[2], Number((e.currentTarget as HTMLInputElement).value)],
                });
              }}
            />
            <output for="twilight-opacity">{() => (twilight()?.color[3] ?? 0.95).toFixed(2)}</output>
          </div>
          <div>
            <label for="twilight-attenuation">Attenuation</label>
            <input
              id="twilight-attenuation"
              type="range"
              min="0.1"
              max="4"
              step="0.05"
              value={() => twilight()?.attenuation ?? 0.5}
              disabled={() => cameraMode() || twilight() === null}
              oninput={(e: Event) =>
                updateTwilight({ attenuation: Number((e.currentTarget as HTMLInputElement).value) })
              }
            />
            <output for="twilight-attenuation">{() => (twilight()?.attenuation ?? 0.5).toFixed(2)}</output>
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
              disabled={allSky}
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
              disabled={allSky}
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
