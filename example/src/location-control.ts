import { LngLat, LngLatBounds, Marker, type IControl } from 'maplibre-gl';

export function locationControl(): IControl {
  let remove: (() => void) | undefined;

  return {
    onAdd(map) {
      const group = document.createElement('div');
      group.className = 'maplibregl-ctrl maplibregl-ctrl-group';
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'maplibregl-ctrl-geolocate';
      const geolocation = navigator.geolocation;
      button.disabled = !window.isSecureContext || !geolocation;
      const label = button.disabled ? 'Location not available' : 'Show my location';
      button.title = label;
      button.setAttribute('aria-label', label);
      button.setAttribute('aria-pressed', 'false');
      const icon = document.createElement('span');
      icon.className = 'maplibregl-ctrl-icon';
      icon.setAttribute('aria-hidden', 'true');
      button.append(icon);
      group.append(button);

      let activeRequest: symbol | undefined;
      let clearMarkers: (() => void) | undefined;
      const turnOff = () => {
        activeRequest = undefined;
        button.setAttribute('aria-pressed', 'false');
        button.classList.remove('maplibregl-ctrl-geolocate-waiting');
        clearMarkers?.();
        clearMarkers = undefined;
      };

      button.addEventListener('click', () => {
        if (activeRequest) {
          turnOff();
          return;
        }

        const request = Symbol();
        activeRequest = request;
        button.title = label;
        button.setAttribute('aria-pressed', 'true');
        button.classList.add('maplibregl-ctrl-geolocate-waiting');
        geolocation.getCurrentPosition(
          ({ coords: { longitude, latitude, accuracy } }) => {
            if (activeRequest !== request) return;
            button.classList.remove('maplibregl-ctrl-geolocate-waiting');

            const center = new LngLat(longitude, latitude);
            const circle = document.createElement('div');
            circle.className = 'maplibregl-user-location-accuracy-circle';
            const dot = document.createElement('div');
            dot.className = 'maplibregl-user-location-dot';
            const accuracyMarker = new Marker({ element: circle, pitchAlignment: 'map' }).setLngLat(center).addTo(map);
            const locationMarker = new Marker({ element: dot }).setLngLat(center).addTo(map);
            const resizeCircle = () => {
              const point = map.project(center);
              const meters = center.distanceTo(map.unproject([point.x + 100, point.y]));
              const diameter = (200 * accuracy) / meters;
              if (Number.isFinite(diameter)) circle.style.width = circle.style.height = `${diameter}px`;
            };
            clearMarkers = () => {
              map.off('move', resizeCircle);
              accuracyMarker.remove();
              locationMarker.remove();
            };
            map.on('move', resizeCircle);
            resizeCircle();
            map.fitBounds(LngLatBounds.fromLngLat(center, accuracy), { maxZoom: 15, bearing: map.getBearing() });
          },
          () => {
            if (activeRequest !== request) return;
            turnOff();
            button.title = 'Location not available';
          },
          { timeout: 6000 },
        );
      });

      remove = () => {
        turnOff();
        group.remove();
      };
      return group;
    },
    onRemove() {
      remove?.();
      remove = undefined;
    },
  };
}
