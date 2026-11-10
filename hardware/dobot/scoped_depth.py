"""Local optical depth correction; raw camera recordings are never modified."""
import numpy as np

DEPTH_PROCESSING_REVISION = '20261021-retain-raw-outside-fit-v2'


def depth_diagnostic(raw_m, meta, row, col, radius=8):
    """Explain a failed sample without replacing it or guessing object identity."""
    raw = np.asarray(raw_m, dtype=float)
    if (raw.ndim != 2 or type(row) is not int or type(col) is not int
            or not 0 <= row < raw.shape[0] or not 0 <= col < raw.shape[1]):
        return dict(reason='pixel_out_of_bounds', image_size=list(raw.shape))
    corrected = corrected_depth(raw, meta)
    value = float(raw[row, col])
    corrected_value = float(corrected[row, col])
    if not np.isfinite(value) or value <= 0:
        reason = 'raw_depth_missing'
    elif value >= 10:
        reason = 'raw_depth_out_of_sensor_range'
    elif (meta.get('local_depth_correction') and not
          meta['local_depth_correction']['valid_raw_depth_range_m'][0] <= value <=
          meta['local_depth_correction']['valid_raw_depth_range_m'][1]):
        reason = 'valid_uncorrected_raw_depth'
    else:
        reason = 'valid_depth_but_projection_unavailable'
    r0, r1 = max(0, row-radius), min(raw.shape[0], row+radius+1)
    c0, c1 = max(0, col-radius), min(raw.shape[1], col+radius+1)
    patch = corrected[r0:r1, c0:c1]
    rr, cc = np.nonzero(np.isfinite(patch) & (patch > 0))
    order = np.argsort((rr+r0-row)**2+(cc+c0-col)**2)
    candidates = [dict(row=int(rr[i]+r0), col=int(cc[i]+c0),
                       depth_m=float(patch[rr[i], cc[i]])) for i in order[:6]]
    return dict(reason=reason, raw_depth_m=value if np.isfinite(value) else None,
                corrected_depth_m=corrected_value, patch_radius_px=radius,
                valid_pixels_in_patch=len(rr), patch_pixels=int(patch.size),
                nearby_measured_pixels=candidates,
                guidance='These pixels may be a different surface. Check RGB object membership before selecting one. No depth was filled or extrapolated.')


def projection_intrinsics_valid(reference, current):
    """Validate the current SDK optical mode, not drift from a focal fit.

    Projection uses current intrinsics. Historical focal/centre differences are
    diagnostics, not a reason to disable calibrated native-depth projection.
    Depth corrections retain their separate fitted-camera binding below.
    """
    if any(reference.get(k) != current.get(k) for k in ('width', 'height', 'model', 'coeffs')):
        return False
    try:
        values = [current[k] for k in ('fx', 'fy', 'ppx', 'ppy')]
        return bool(np.isfinite(values).all() and min(values[:2]) > 0
                    and current['width'] > 0 and current['height'] > 0
                    and 0 <= current['ppx'] < current['width']
                    and 0 <= current['ppy'] < current['height'])
    except (KeyError, TypeError, ValueError):
        return False


def intrinsics_match(reference, current, focal_range_px=None, validated_focal_span_px=None,
                     validated_focal_margin_px=None):
    """Use a measured focal envelope consistently for geometry and depth fits."""
    if any(reference.get(k) != current.get(k) for k in ('width', 'height', 'model', 'coeffs')):
        return False
    try:
        values = [current[k] for k in ('fx', 'fy', 'ppx', 'ppy')]
        if not np.isfinite(values).all() or min(values[:2]) <= 0:
            return False
        if validated_focal_span_px is not None and not all(
                np.isfinite(validated_focal_span_px[k]) and validated_focal_span_px[k] >= 0
                for k in ('fx', 'fy')):
            return False
        if focal_range_px is not None:
            # New calibrations record a margin checked against SDK ray sensitivity.
            # A single observed focal value must not imply zero optical tolerance.
            margin = .05 if validated_focal_margin_px is None else validated_focal_margin_px
            if not np.isfinite(margin) or not 0 <= margin <= 1.:
                return False
            focal_ok = all(
                len(focal_range_px[k]) == 2
                and np.isfinite(focal_range_px[k]).all()
                and 0 <= focal_range_px[k][1] - focal_range_px[k][0] <= (
                    validated_focal_span_px[k] if validated_focal_span_px is not None else 1.)
                and focal_range_px[k][0] - margin <= current[k] <= focal_range_px[k][1] + margin
                for k in ('fx', 'fy'))
        else:
            focal_ok = np.allclose([reference[k] for k in ('fx', 'fy')],
                                   values[:2], rtol=0, atol=.25)
        return bool(focal_ok and np.allclose(
            [reference[k] for k in ('ppx', 'ppy')], values[2:], rtol=0, atol=.25))
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def binding_matches(binding, meta):
    """A depth fit belongs to one camera, optical mode and depth unit."""
    if not binding or binding.get('serial_number') != meta.get('serial_number'):
        return False
    try:
        return bool(intrinsics_match(binding.get('intrinsics', {}), meta.get('intrinsics', {}),
                                     binding.get('focal_range_px'), binding.get('validated_focal_span_px'),
                                     binding.get('validated_focal_margin_px'))
            and np.isclose(binding['depth_scale_m'], meta['depth_scale_m'], rtol=1e-9, atol=0))
    except (KeyError, TypeError, ValueError):
        return False


def corrected_depth(raw_m, meta):
    raw = np.asarray(raw_m, dtype=float)
    valid = np.isfinite(raw) & (raw > 0) & (raw < 10)
    correction = meta.get('local_depth_correction')
    if correction is None:
        return np.where(valid, raw, 0.)
    if correction.get('local_depth_validation_passed') is not True:
        raise ValueError('Unvalidated local depth correction')
    if not binding_matches(correction.get('camera_binding'), meta):
        raise ValueError('Depth correction camera/mode/scale binding mismatch; refit required')
    low, high = correction['valid_raw_depth_range_m']
    offset = correction['offset_m']
    model=correction.get('model','offset')
    scale=correction.get('scale',1.) if model=='affine' else 1.
    if (model not in ('offset','affine') or not np.isfinite([low,high,offset,scale]).all()
        or not 0<low<high or not .9<scale<1.1
        or max(abs((scale-1)*low+offset),abs((scale-1)*high+offset))>=.04):
        raise ValueError('Invalid local depth correction')
    scoped = valid & (raw >= low) & (raw <= high)
    # A correction's measured interval is not the sensor's usable range.
    # Preserve valid raw samples outside it for BOTH views, without extending
    # the fitted correction or fabricating missing sensor measurements.
    result = np.where(scoped, raw*scale+offset, raw)
    return np.where(valid & (result > 0), result, 0.)
