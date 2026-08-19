# Flickr30k to COCO migration record

## Reason

The official Flickr30k page remained reachable, but its linked official Box image archive returned HTTP 404. The project therefore stopped treating Flickr30k as a required primary dataset. It did not use an unofficial mirror, scraping, or synthetic replacement.

## New primary scope

The primary dataset is `coco2017_val`: the complete official COCO 2017 `val2017` image release plus `captions_val2017.json`. The official download page and terms page were checked before download. Archive sizes and hashes are stored in the manifest metadata and Phase 1 artifacts.

## Compatibility contract

The common dataset layer now represents an image group with stable image ID and filename, source URL and dataset/version metadata, caption records with stable IDs, original and conservatively normalized text, split assignment, and validation/tier provenance.

Acquisition is dataset-specific (`acquisition.py` for historical Flickr30k and `coco_acquisition.py` for official COCO), while validation, image integrity, splitting, leakage checking, statistics, and experiment inputs are common.

## Preserved history

Historical Flickr30k manifests, audits, and artifacts are retained. They are not rewritten as COCO and are not used to invent COCO counts. The old official Flickr metadata remains useful for future external validation if an authorized image archive is obtained.

## Re-executed evidence

COCO Phase 1–6 artifacts are separated under `artifacts/coco/`; the real runs include image validation, image EDA, a classical image descriptor, frozen CLIP cross-modal evaluation, canonical Phase 5 migration, and real frozen vision representation evidence. Precise executed scopes and limitations are in the corresponding JSON reports.
