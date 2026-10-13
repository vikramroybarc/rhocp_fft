#!/usr/bin/env python3
"""Build DDCP orientation.in from a grain Euler table and a voxel grain-ID map.

Input grain Euler table: four columns: grain_id phi1 Phi phi2, degrees.
Input voxel grain map: any whitespace-separated integer IDs, z fastest.
# comments are permitted in both inputs.
"""
import argparse
import math
from pathlib import Path


def rows(path):
    for line_number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split('#', 1)[0].strip()
        if line:
            yield line_number, line.split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx', required=True, type=int)
    parser.add_argument('--ny', required=True, type=int)
    parser.add_argument('--nz', required=True, type=int)
    parser.add_argument('--orientations', required=True, help='grain_id phi1 Phi phi2 table')
    parser.add_argument('--voxel-ids', required=True, help='voxel IDs, z fastest')
    parser.add_argument('--output', default='orientation.in')
    args = parser.parse_args()

    dims = (args.nx, args.ny, args.nz)
    if min(dims) <= 0:
        parser.error('GRID sizes must be positive')
    grains = {}
    for lineno, values in rows(args.orientations):
        if len(values) != 4:
            parser.error(f'{args.orientations}:{lineno}: expected grain_id phi1 Phi phi2')
        gid = int(values[0])
        if gid <= 0 or gid in grains:
            parser.error(f'{args.orientations}:{lineno}: invalid/duplicate grain ID {gid}')
        angles = tuple(map(float, values[1:]))
        if not all(math.isfinite(x) for x in angles):
            parser.error(f'{args.orientations}:{lineno}: Euler angles must be finite')
        grains[gid] = angles

    voxel_ids = []
    for lineno, values in rows(args.voxel_ids):
        try:
            voxel_ids.extend(map(int, values))
        except ValueError:
            parser.error(f'{args.voxel_ids}:{lineno}: grain IDs must be integers')
    expected = math.prod(dims)
    if len(voxel_ids) != expected:
        parser.error(f'expected {expected} voxel IDs, found {len(voxel_ids)}')
    missing = set(voxel_ids) - set(grains)
    if missing:
        parser.error(f'voxel map uses undefined grain IDs: {sorted(missing)}')

    lines = [f'GRID {args.nx} {args.ny} {args.nz}', f'GRAINS {len(grains)}']
    lines.extend(f'{gid} {e[0]:.16g} {e[1]:.16g} {e[2]:.16g}' for gid, e in sorted(grains.items()))
    lines.append('VOXELS')
    for i in range(args.nx):
        for j in range(args.ny):
            begin = (i * args.ny + j) * args.nz
            lines.append(' '.join(map(str, voxel_ids[begin:begin + args.nz])))
    Path(args.output).write_text('\n'.join(lines) + '\n')
    print(f'Wrote {args.output}: {len(grains)} grains, {expected} voxels')


if __name__ == '__main__':
    main()
