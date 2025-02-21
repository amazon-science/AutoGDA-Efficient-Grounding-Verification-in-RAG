_BUCKET_NAME = "factual-consistency"
_SD_BUCKET_NAME = "factual-consistency-synthetic-data"
NLI_LABELS = {"contradiction": 0, "entailment": 1, "neutral": 2}
INV_NLI_LABELS = {v: k for (k, v) in NLI_LABELS.items()}

class COLOR:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    GRAY = '\033[90m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'