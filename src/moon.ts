/** Low-precision lunar ephemeris (J2000-era orbital elements and main perturbations).
 * Uses the orbital elements and perturbations described by Paul Schlyter:
 * https://stjarnhimlen.se/comp/ppcomp.html
 * Intended for illumination, not occultations; no refraction or libration.
 */
export function lunarLight(date: Date): { position: number[]; phaseAngle: number; strength: number } {
  const rad = Math.PI / 180;
  const sin = (angle: number) => Math.sin(angle * rad);
  const cos = (angle: number) => Math.cos(angle * rad);
  const days = date.getTime() / 86400000 - 10956;
  const node = 125.1228 - 0.0529538083 * days;
  const inclination = 5.1454;
  const periapsis = 318.0634 + 0.1643573223 * days;
  const anomaly = 115.3654 + 13.0649929509 * days;
  const eccentricity = 0.0549;
  let eccentric = anomaly * rad;
  for (let i = 0; i < 8; i++)
    eccentric -=
      (eccentric - eccentricity * Math.sin(eccentric) - anomaly * rad) / (1 - eccentricity * Math.cos(eccentric));
  const x = Math.cos(eccentric) - eccentricity;
  const y = Math.sqrt(1 - eccentricity ** 2) * Math.sin(eccentric);
  const trueAnomaly = Math.atan2(y, x) / rad;
  let distance = 60.2666 * Math.hypot(x, y);
  const argument = trueAnomaly + periapsis;
  const ecliptic = [
    cos(node) * cos(argument) - sin(node) * sin(argument) * cos(inclination),
    sin(node) * cos(argument) + cos(node) * sin(argument) * cos(inclination),
    sin(argument) * sin(inclination),
  ];
  let longitude = Math.atan2(ecliptic[1], ecliptic[0]) / rad;
  let latitude = Math.asin(ecliptic[2]) / rad;
  const sunAnomaly = 356.047 + 0.9856002585 * days;
  const sunPeriapsis = 282.9404 + 0.0000470935 * days;
  const sunLongitude = sunAnomaly + sunPeriapsis;
  const meanLongitude = anomaly + periapsis + node;
  const elongation = meanLongitude - sunLongitude;
  const argumentLatitude = meanLongitude - node;
  longitude +=
    -1.274 * sin(anomaly - 2 * elongation) +
    0.658 * sin(2 * elongation) -
    0.186 * sin(sunAnomaly) -
    0.059 * sin(2 * anomaly - 2 * elongation) -
    0.057 * sin(anomaly - 2 * elongation + sunAnomaly) +
    0.053 * sin(anomaly + 2 * elongation) +
    0.046 * sin(2 * elongation - sunAnomaly) +
    0.041 * sin(anomaly - sunAnomaly) -
    0.035 * sin(elongation) -
    0.031 * sin(anomaly + sunAnomaly) -
    0.015 * sin(2 * argumentLatitude - 2 * elongation) +
    0.011 * sin(anomaly - 4 * elongation);
  latitude +=
    -0.173 * sin(argumentLatitude - 2 * elongation) -
    0.055 * sin(anomaly - argumentLatitude - 2 * elongation) -
    0.046 * sin(anomaly + argumentLatitude - 2 * elongation) +
    0.033 * sin(argumentLatitude + 2 * elongation) +
    0.017 * sin(2 * anomaly + argumentLatitude);
  distance += -0.58 * cos(anomaly - 2 * elongation) - 0.46 * cos(2 * elongation);
  const obliquity = 23.4393 - 3.563e-7 * days;
  const ex = cos(longitude) * cos(latitude);
  const ey = sin(longitude) * cos(latitude);
  const ez = sin(latitude);
  const equatorial = [ex, ey * cos(obliquity) - ez * sin(obliquity), ey * sin(obliquity) + ez * cos(obliquity)];
  const j2000Days = days - 1.5;
  const t = j2000Days / 36525;
  const sidereal = 280.46061837 + 360.98564736629 * j2000Days + 0.000387933 * t * t - t ** 3 / 38710000;
  const kilometres = distance * 6378.14;
  const position = [
    (equatorial[1] * cos(sidereal) - equatorial[0] * sin(sidereal)) * kilometres,
    equatorial[2] * kilometres,
    (equatorial[0] * cos(sidereal) + equatorial[1] * sin(sidereal)) * kilometres,
  ];
  const sunTrueLongitude = sunLongitude + 1.915 * sin(sunAnomaly) + 0.02 * sin(2 * sunAnomaly);
  const sunDistance = 149597870.7 * (1 - 0.01671 * cos(sunAnomaly));
  const separationCos = cos(latitude) * cos(longitude - sunTrueLongitude);
  const separationSin = Math.sqrt(Math.max(0, 1 - separationCos ** 2));
  const phaseAngle = Math.atan2(sunDistance * separationSin, kilometres - sunDistance * separationCos);
  const phaseDegrees = phaseAngle / rad;
  // Empirical visual phase law (Krisciunas & Schaefer 1991); full-Moon
  // normal illuminance is approximately 0.25 lux at the mean lunar distance.
  // The empirical fit is not valid at new Moon. Fade its extrapolated tail
  // over the final ten degrees instead of assigning finite light to a dark disk.
  const crescent = Math.max(0, Math.min(1, (180 - phaseDegrees) / 10));
  const strength =
    0.25 *
    (384400 / kilometres) ** 2 *
    10 ** (-0.4 * (0.026 * phaseDegrees + 4e-9 * phaseDegrees ** 4)) *
    crescent *
    crescent *
    (3 - 2 * crescent);
  return { position, phaseAngle, strength };
}
