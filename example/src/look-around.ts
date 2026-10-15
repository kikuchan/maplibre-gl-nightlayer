import {
  LngLat,
  MercatorCoordinate,
  type CameraOptions,
  type CustomLayerInterface,
  type CustomRenderMethodInput,
  type Map,
  type ProjectionSpecification,
} from 'maplibre-gl';

const radians = Math.PI / 180;
type Direction = [number, number, number];
type DragMode = 'look' | 'move';
const dot = (a: Direction, b: Direction): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export class LookAround implements CustomLayerInterface {
  id = 'look-around-controls';
  type = 'custom' as const;
  #map: Map;
  #onChange: (available: boolean, active: boolean) => void;
  #available = false;
  #observer?: MercatorCoordinate;
  #distance = 0;
  #movementMetersPerPixel = 0;
  #heightLimit = 0;
  #pitch = 0;
  #bearing = 0;
  #roll = 0;
  #pointer?: {
    id: number;
    time: number;
    anchor: Direction;
    x: number;
    y: number;
    mode: DragMode;
  };
  #velocity = { bearing: 0, pitch: 0, right: 0, forward: 0 };
  #inertiaFrame?: number;
  #saved?: {
    camera: CameraOptions;
    projection: ProjectionSpecification;
    maxPitch: number;
    clamped: boolean;
    cursor: string;
    touchAction: string;
    handlers: { handler: { enable(): void; disable(): void }; enabled: boolean }[];
  };

  constructor(map: Map, onChange: (available: boolean, active: boolean) => void) {
    this.#map = map;
    this.#onChange = onChange;
  }

  // Keep temporary look-around gestures out of the normal map's shared camera state.
  get returnView(): CameraOptions | undefined {
    return this.#saved?.camera;
  }

  render(_gl: WebGLRenderingContext | WebGL2RenderingContext, options: CustomRenderMethodInput): void {
    const available = options.defaultProjectionData.projectionTransition === 0 && this.#map.getZoom() >= 12;
    if (available !== this.#available) {
      this.#available = available;
      this.#onChange(available, !!this.#saved);
    }
  }

  toggle(): void {
    if (this.#saved) this.stop();
    else this.#start();
  }

  setFieldOfView(value: number): void {
    this.#map.setVerticalFieldOfView(value);
    // The transform callback runs before the new FOV becomes the map's current
    // FOV. Recalculate the camera afterwards to keep the observer stationary.
    if (this.#saved) this.#map.jumpTo(this.#cameraOptions());
  }

  #start(): void {
    if (!this.#available || this.#map.getZoom() < 12) return;
    const map = this.#map;
    map.stop();
    const canvas = map.getCanvas();
    const center = MercatorCoordinate.fromLngLat(map.getCenter(), map.getCenterElevation());
    const distance =
      canvas.clientHeight / (2 * Math.tan((map.getVerticalFieldOfView() * radians) / 2)) / (512 * 2 ** map.getZoom());
    this.#pitch = map.getPitch();
    this.#bearing = map.getBearing();
    this.#roll = map.getRoll();
    this.#observer = new MercatorCoordinate(
      center.x - distance * Math.sin(this.#pitch * radians) * Math.sin(this.#bearing * radians),
      center.y + distance * Math.sin(this.#pitch * radians) * Math.cos(this.#bearing * radians),
      center.z + distance * Math.cos(this.#pitch * radians),
    );
    this.#distance = distance;
    this.#movementMetersPerPixel =
      (2 * distance * Math.tan((map.getVerticalFieldOfView() * radians) / 2)) /
      (canvas.clientHeight * this.#observer.meterInMercatorCoordinateUnits());
    const observerGround = map.queryTerrainElevation(this.#observer.toLngLat()) ?? 0;
    // A ground-facing camera at zoom 12 is our conservative Mercator-only
    // boundary. Keep the entry view's pitch/FOV as the reference for the limit.
    this.#heightLimit = Math.max(
      this.#observer.toAltitude() - observerGround,
      (distance * 2 ** (map.getZoom() - 12) * Math.cos(this.#pitch * radians)) /
        this.#observer.meterInMercatorCoordinateUnits(),
    );
    const handlers = [
      map.dragPan,
      map.dragRotate,
      map.scrollZoom,
      map.boxZoom,
      map.doubleClickZoom,
      map.keyboard,
      map.touchZoomRotate,
      map.touchPitch,
    ];
    this.#saved = {
      camera: {
        center: map.getCenter(),
        zoom: map.getZoom(),
        pitch: map.getPitch(),
        bearing: map.getBearing(),
        roll: map.getRoll(),
        elevation: map.getCenterElevation(),
      },
      projection: map.getProjection(),
      maxPitch: map.getMaxPitch(),
      clamped: map.getCenterClampedToGround(),
      cursor: canvas.style.cursor,
      touchAction: canvas.style.touchAction,
      handlers: handlers.map((handler) => ({ handler, enabled: handler.isEnabled() })),
    };
    for (const { handler } of this.#saved.handlers) handler.disable();
    map.setProjection({ type: 'mercator' });
    map.setCenterClampedToGround(false);
    map.setMaxPitch(180);
    // Also prevent navigation controls or other camera commands from moving the observer.
    map.setTransformCameraUpdate(() => this.#cameraOptions());
    canvas.style.cursor = 'grab';
    canvas.style.touchAction = 'none';
    canvas.addEventListener('pointerdown', this.#pointerDown);
    canvas.addEventListener('pointermove', this.#pointerMove);
    canvas.addEventListener('pointerup', this.#pointerUp);
    canvas.addEventListener('pointercancel', this.#pointerUp);
    canvas.addEventListener('lostpointercapture', this.#pointerUp);
    canvas.addEventListener('wheel', this.#wheel, { passive: false });
    canvas.addEventListener('keydown', this.#keyDown);
    this.#onChange(true, true);
    map.jumpTo(this.#cameraOptions());
  }

  #cameraOptions(distance = this.#distance) {
    const camera = this.#observer!;
    const target = new MercatorCoordinate(
      camera.x + distance * Math.sin(this.#pitch * radians) * Math.sin(this.#bearing * radians),
      camera.y - distance * Math.sin(this.#pitch * radians) * Math.cos(this.#bearing * radians),
      camera.z - distance * Math.cos(this.#pitch * radians),
    );
    const options = this.#map.calculateCameraOptionsFromTo(
      camera.toLngLat(),
      camera.toAltitude(),
      target.toLngLat(),
      target.toAltitude(),
    );
    return {
      ...options,
      center: LngLat.convert(options.center!),
      // A vertical ray has no horizontal direction. Preserve our orientation
      // instead of inferring it from coincident horizontal camera/target points.
      bearing: this.#bearing,
      pitch: this.#pitch,
      roll: this.#roll,
    };
  }

  #returnCamera(maxPitch: number, clamped: boolean): CameraOptions {
    this.#pitch = Math.min(this.#pitch, maxPitch);
    if (!clamped) return this.#cameraOptions();

    // Move the target along the current ray to the ground, rather than moving the
    // observer back to the old map center. Terrain heights are best-effort: tiles
    // outside the loaded area may not yet be available.
    const observer = this.#observer!;
    const vertical = Math.cos(this.#pitch * radians);
    let distance = this.#distance;
    for (let i = 0; i < 12; i++) {
      const options = this.#cameraOptions(distance);
      const ground = this.#map.queryTerrainElevation(options.center) ?? 0;
      const height = MercatorCoordinate.fromLngLat(options.center, ground).z;
      const next = (observer.z - height) / vertical;
      if (next <= 0 || !Number.isFinite(next)) break;
      const converged = Math.abs(next - distance) < 1e-10;
      distance = next;
      if (converged) break;
    }
    return this.#cameraOptions(distance);
  }

  #screenRay(x: number, y: number, bearing = this.#bearing, pitch = this.#pitch): Direction {
    const canvas = this.#map.getCanvas();
    const bounds = canvas.getBoundingClientRect();
    const scale = (2 * Math.tan((this.#map.getVerticalFieldOfView() * radians) / 2)) / bounds.height;
    const screenX = (x - bounds.left - bounds.width / 2) * scale;
    const screenY = (bounds.top + bounds.height / 2 - y) * scale;
    const b = bearing * radians,
      p = pitch * radians,
      r = this.#roll * radians;
    const right: Direction = [Math.cos(b), Math.sin(b), 0];
    const up: Direction = [Math.cos(p) * Math.sin(b), -Math.cos(p) * Math.cos(b), Math.sin(p)];
    const forward: Direction = [Math.sin(p) * Math.sin(b), -Math.sin(p) * Math.cos(b), -Math.cos(p)];
    const horizontal = screenX * Math.cos(r) + screenY * Math.sin(r);
    const vertical = -screenX * Math.sin(r) + screenY * Math.cos(r);
    const ray = forward.map((value, i) => value + horizontal * right[i] + vertical * up[i]) as Direction;
    const length = Math.hypot(...ray);
    return ray.map((value) => value / length) as Direction;
  }

  #followPointer(x: number, y: number, anchor: Direction): void {
    // Solve for the orientation that keeps the grabbed ray under the pointer.
    // Using screen rays makes the gesture independent of field of view and
    // includes roll. Pitch limits can prevent an exact match at the poles.
    const step = 0.001;
    for (let i = 0; i < 12; i++) {
      const ray = this.#screenRay(x, y);
      const error = anchor.map((value, j) => value - ray[j]) as Direction;
      if (Math.hypot(...error) < 1e-9) break;
      const db = this.#screenRay(x, y, this.#bearing + step).map((value, j) => (value - ray[j]) / step) as Direction;
      const dp = this.#screenRay(x, y, this.#bearing, this.#pitch + step).map(
        (value, j) => (value - ray[j]) / step,
      ) as Direction;
      const bb = dot(db, db) + 1e-12,
        pp = dot(dp, dp) + 1e-12,
        bp = dot(db, dp);
      const be = dot(db, error),
        pe = dot(dp, error),
        determinant = bb * pp - bp * bp;
      const bearing = (be * pp - pe * bp) / determinant;
      const pitch = (pe * bb - be * bp) / determinant;
      this.#bearing += Math.max(-30, Math.min(30, bearing));
      this.#pitch = Math.max(0, Math.min(180, this.#pitch + Math.max(-30, Math.min(30, pitch))));
    }
  }

  #pointerDown = (event: PointerEvent): void => {
    if (this.#pointer || (event.pointerType === 'mouse' && event.button !== 0 && event.button !== 2)) return;
    event.preventDefault();
    this.#stopInertia();
    const canvas = this.#map.getCanvas();
    canvas.focus({ preventScroll: true });
    canvas.setPointerCapture(event.pointerId);
    canvas.style.cursor = 'grabbing';
    this.#pointer = {
      id: event.pointerId,
      time: event.timeStamp,
      anchor: this.#screenRay(event.clientX, event.clientY),
      x: event.clientX,
      y: event.clientY,
      mode: this.#dragMode(event),
    };
  };

  #pointerMove = (event: PointerEvent): void => {
    const pointer = this.#pointer;
    if (!pointer || pointer.id !== event.pointerId) return;
    event.preventDefault();
    const mode = this.#dragMode(event);
    if (mode !== pointer.mode) {
      this.#stopInertia();
      pointer.anchor = this.#screenRay(pointer.x, pointer.y);
    }
    const elapsed = Math.max(1, event.timeStamp - pointer.time);
    const weight = 1 - Math.exp(-elapsed / 40);
    if (mode !== 'look') {
      const right = (pointer.x - event.clientX) * this.#movementMetersPerPixel;
      const forward = (event.clientY - pointer.y) * this.#movementMetersPerPixel;
      this.#moveObserver(right, forward);
      this.#velocity.right += (right / elapsed - this.#velocity.right) * weight;
      this.#velocity.forward += (forward / elapsed - this.#velocity.forward) * weight;
    } else {
      const previousBearing = this.#bearing,
        previousPitch = this.#pitch;
      this.#followPointer(event.clientX, event.clientY, pointer.anchor);
      const bearing = this.#bearing - previousBearing;
      const pitch = this.#pitch - previousPitch;
      this.#velocity.bearing += (Math.max(-0.5, Math.min(0.5, bearing / elapsed)) - this.#velocity.bearing) * weight;
      this.#velocity.pitch += (Math.max(-0.5, Math.min(0.5, pitch / elapsed)) - this.#velocity.pitch) * weight;
    }
    this.#pointer = { ...pointer, x: event.clientX, y: event.clientY, mode, time: event.timeStamp };
    this.#map.jumpTo(this.#cameraOptions());
  };

  #pointerUp = (event: PointerEvent): void => {
    if (this.#pointer?.id !== event.pointerId) return;
    const coast = event.type === 'pointerup' && event.timeStamp - this.#pointer.time < 100;
    this.#pointer = undefined;
    const canvas = this.#map.getCanvas();
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    canvas.style.cursor = 'grab';
    if (coast) this.#startInertia();
    else this.#stopInertia();
  };

  #stopInertia(): void {
    if (this.#inertiaFrame !== undefined) cancelAnimationFrame(this.#inertiaFrame);
    this.#inertiaFrame = undefined;
    this.#velocity = { bearing: 0, pitch: 0, right: 0, forward: 0 };
  }

  #dragMode(event: PointerEvent): DragMode {
    return event.shiftKey ? 'move' : 'look';
  }

  #moveObserver(right: number, forward: number, height = 0): void {
    const observer = this.#observer!;
    const altitude = observer.toAltitude() + height;
    const scale = observer.meterInMercatorCoordinateUnits();
    const b = this.#bearing * radians;
    const position = new MercatorCoordinate(
      observer.x + (right * Math.cos(b) + forward * Math.sin(b)) * scale,
      observer.y + (right * Math.sin(b) - forward * Math.cos(b)) * scale,
    ).toLngLat();
    // Preserve sea-level altitude when changing latitude, and avoid descending
    // into known terrain. Heights outside loaded terrain fall back to sea level.
    position.lat = Math.max(-85.051129, Math.min(85.051129, position.lat));
    const ground = this.#map.queryTerrainElevation(position) ?? 0;
    this.#observer = MercatorCoordinate.fromLngLat(
      position,
      Math.max(ground + 1, Math.min(ground + this.#heightLimit, altitude)),
    );
  }

  #startInertia(): void {
    let previous = performance.now();
    const frame = (now: number) => {
      const elapsed = Math.min(now - previous, 64);
      previous = now;
      const decay = Math.exp(-elapsed / 250);
      const distance = 250 * (1 - decay);
      this.#bearing += this.#velocity.bearing * distance;
      this.#pitch = Math.max(0, Math.min(180, this.#pitch + this.#velocity.pitch * distance));
      this.#velocity.bearing *= decay;
      this.#velocity.pitch *= decay;
      if (this.#pitch === 0 || this.#pitch === 180) this.#velocity.pitch = 0;
      if (this.#velocity.right || this.#velocity.forward) {
        this.#moveObserver(this.#velocity.right * distance, this.#velocity.forward * distance);
        this.#velocity.right *= decay;
        this.#velocity.forward *= decay;
      }
      this.#map.jumpTo(this.#cameraOptions());
      if (
        Math.hypot(this.#velocity.bearing, this.#velocity.pitch) > 0.002 ||
        Math.hypot(this.#velocity.right, this.#velocity.forward) > 0.0001
      ) {
        this.#inertiaFrame = requestAnimationFrame(frame);
      } else this.#stopInertia();
    };
    this.#inertiaFrame = requestAnimationFrame(frame);
  }

  #wheel = (event: WheelEvent): void => {
    event.preventDefault();
    this.#stopInertia();
    const pixels =
      event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? this.#map.getCanvas().clientHeight : 1);
    const observer = this.#observer!;
    const ground = this.#map.queryTerrainElevation(observer.toLngLat()) ?? 0;
    const clearance = Math.max(1, observer.toAltitude() - ground);
    // Scale speed by ground clearance, not FOV: a telephoto lens should not
    // make physical movement impossibly slow. Limit individual wheel jumps.
    const distance = clearance * Math.max(-0.5, Math.min(0.5, -pixels * 0.002));
    if (event.shiftKey) this.#moveObserver(0, 0, distance);
    else this.#moveObserver(0, distance * Math.sin(this.#pitch * radians), -distance * Math.cos(this.#pitch * radians));
    if (this.#pointer) {
      this.#pointer.anchor = this.#screenRay(this.#pointer.x, this.#pointer.y);
    }
    this.#map.jumpTo(this.#cameraOptions());
  };

  #keyDown = (event: KeyboardEvent): void => {
    if (event.key === 'Escape') {
      event.preventDefault();
      this.stop();
      return;
    }
    if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault();
    this.#stopInertia();
    const step = event.shiftKey ? 10 : 2;
    if (event.key === 'ArrowUp') this.#pitch += step;
    if (event.key === 'ArrowDown') this.#pitch -= step;
    if (event.key === 'ArrowLeft') this.#bearing -= step;
    if (event.key === 'ArrowRight') this.#bearing += step;
    this.#pitch = Math.max(0, Math.min(180, this.#pitch));
    this.#map.jumpTo(this.#cameraOptions());
  };

  stop(restore = true): void {
    const saved = this.#saved;
    if (!saved) return;
    this.#stopInertia();
    this.#saved = undefined;
    const map = this.#map,
      canvas = map.getCanvas();
    canvas.removeEventListener('pointerdown', this.#pointerDown);
    canvas.removeEventListener('pointermove', this.#pointerMove);
    canvas.removeEventListener('pointerup', this.#pointerUp);
    canvas.removeEventListener('pointercancel', this.#pointerUp);
    canvas.removeEventListener('lostpointercapture', this.#pointerUp);
    canvas.removeEventListener('wheel', this.#wheel);
    canvas.removeEventListener('keydown', this.#keyDown);
    if (this.#pointer && canvas.hasPointerCapture(this.#pointer.id)) canvas.releasePointerCapture(this.#pointer.id);
    this.#pointer = undefined;
    canvas.style.cursor = saved.cursor;
    canvas.style.touchAction = saved.touchAction;
    map.setTransformCameraUpdate(null);
    if (restore) {
      const camera = this.#returnCamera(saved.maxPitch, saved.clamped);
      map.setCenterClampedToGround(saved.clamped);
      map.jumpTo(camera);
      map.setMaxPitch(saved.maxPitch);
      map.setProjection(saved.projection);
      for (const { handler, enabled } of saved.handlers) if (enabled) handler.enable();
    }
    this.#onChange(this.#available, false);
  }

  onRemove(): void {
    this.stop(false);
  }
}
