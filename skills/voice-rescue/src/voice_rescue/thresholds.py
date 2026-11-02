"""Every threshold the skill decides on, in one place. Where each number comes from is documented in
references/thresholds.md (practice on real recordings and the benchmark in evals/)."""

# "Podcast mic" target, measured with metrics.score() on 60 s speech windows.
TARGET_OVRL = 3.3
TARGET_BAK = 4.0
TARGET_SRMR = None  # not a target: clean studio voices span SRMR 4-17 (see references/thresholds.md)

# Rule "SRMR is not a universal not-worse gate": on an already dry close mic SRMR can drop after
# restoration while quality rises. For inputs at least this dry, SRMR only has to stay above the floor.
DRY_INPUT_SRMR = 10.0
SRMR_FLOOR = 9.0
SRMR_OVERRIDE_OVRL_GAIN = 0.10  # a lower SRMR is not "worse" when OVRL rose at least this much
OVRL_NOT_WORSE_TOL = 0.03  # spread of one window between equivalent runs (chunk alignment): +-0.03

# Router classes (medians over the analysis windows)
FAR_MIC_SRMR = 8.0
FAR_MIC_OVRL = 2.8
LAPTOP_SRMR = 13.0
LAPTOP_OVRL = 3.1
CLEAN_OVRL = 3.3
CLEAN_BAK = 4.0
CLEAN_SRMR = 10.0
PARTS_LU = 1.0
CLIP_RATIO = 1e-4
FLAT_TOPS = 200
NARROW_CUTOFF_HZ = 8600
HUM_PROMINENCE_DB = 10.0

# Deliverable
LOUDNESS_TARGET = -16.0
LOUDNESS_TOL = 0.5
SEGMENT_TOL = 1.0
TRUE_PEAK_MAX = -1.0
SYNC_MS = 1.0
SYNC_POINTS = 9
TIMBRE_REF_DB = 1.0

# Words and voice
WER_MARGIN = 0.05        # allowed WER(input, output) above the recogniser's own noise floor
SPEAKER_MIN = 0.65       # cosine of ECAPA embeddings: two different people reached 0.633, the same person
                         # after restoration never fell below 0.667 (evals/results/thresholds.json)

# Long recordings: probe the hardest stretch first
PROBE_MIN_DURATION = 360.0
PROBE_LEN = 150.0

# Fix loop
MAX_ROUNDS = 3

# Hypothesis branches. Enabled only where the benchmark showed a win over the default path.
EXPERIMENTAL_DEFAULTS = {
    "declip": False,
    "hum_notch": False,
}
