# Data validation and leakage protection

## Active dataset

The project uses official COCO 2017 `val2017` images and official caption
annotations. The local manifest contains 5,000 unique image groups and 25,014
captions. Caption cardinalities are 4,987 groups with five captions, 12 with
six, and one with seven.

Pillow decoded all 5,000 local images. The recorded validation found zero
missing, unreadable, or corrupted images and zero exact byte-duplicate image
groups. Normalized duplicate captions remain warnings rather than being
deleted. Exact image hashing does not detect perceptual near-duplicates or
different files depicting the same scene.

## Split contract

The saved split contains 4,000 train, 500 validation, and 500 test image
groups, with 20,011, 2,502, and 2,501 captions respectively. A SHA-256 order
of stable image IDs with seed 42 selects whole image groups. The leakage
assertion fails if an image ID or caption ID appears in more than one split.
No phase may split caption rows independently.

## Phase 9 mining protections

Phase 9 mines only from the selected train image groups. For each positive
pair it excludes the positive image, every caption owned by the same image,
exact normalized caption aliases, and known exact image-duplicate groups.
The executed run mined 800 pair records and found zero known false negatives
under these checks. Three exact caption aliases were excluded during the
candidate search.

Different image IDs can still be semantically equivalent or near-duplicate;
the active dataset has no complete human semantic-equivalence label. The
false-negative artifact therefore reports a zero known rate separately from
an explicitly non-labeling similarity risk screen.

## Historical Flickr30k decision

The official Flickr30k metadata path was preserved, but the authorized image
archive returned HTTP 404. The project did not use an unofficial mirror or
claim image-backed Flickr30k results. COCO was adopted as the active benchmark
only after its official download path, terms, archive hashes, local image
validation, manifest, and split were verified.
