export const MIN_FOV = 0.1;
export const MAX_FOV = 100;
export const DEFAULT_FOV = 36.86989764584402;

// Example's virtual sensor: a 960 CSS px-high view spans 24 mm.
export const SENSOR_PITCH_MM = 24 / 960;

export function focalLength(fovDegrees: number, sensorHeightMm: number): number {
  return sensorHeightMm / (2 * Math.tan((fovDegrees * Math.PI) / 360));
}
