"""Turn the OpenEP MATLAB exports into the notebook's data bundle.

Run this once, on a machine with the build-time stack installed:

    uv pip install -r requirements-build.txt
    python scripts/build_bundle.py

It downloads and caches the datasets (~67 MB) if they are not already in
`data/`, then writes `public/ep_primer_data.npz`. The notebook — and the
published WebAssembly page — read only that file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epkit.bundle import BUNDLE_NAME, build_bundle, load_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "public" / BUNDLE_NAME,
        help="where to write the bundle (default: public/%(default)s)",
    )
    parser.add_argument(
        "--dataset", type=int, default=1, choices=(1, 2), help="which OpenEP case"
    )
    args = parser.parse_args()

    print(f"building bundle from dataset {args.dataset} ...")
    path = build_bundle(args.output, dataset=args.dataset)
    size_mb = path.stat().st_size / 1e6

    bundle = load_bundle(path)
    print(f"\nwrote {path} ({size_mb:.1f} MB)")
    print(bundle)
    print(f"  vertices        : {len(bundle.points):,}")
    print(f"  mapping points  : {bundle.meta['n_points']:,}")
    print(f"  lesion tags     : {len(bundle.tag_points):,}")
    print(f"  free boundaries : {len(bundle.boundary_rings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
