# from absl import flags
import sys

NLI_LABELS = {"contradiction": 0, "entailment": 1, "neutral": 2}

# _BUCKET_NAME = 'privacy-datasets-llms'
_BUCKET_NAME = 'harpo-hallucination-detection'
_SD_BUCKET_NAME = 'synthetic-nli-data'

_MODEL_NAMES = {}
MODELS = ['tals/albert-xlarge-vitaminc-mnli', 'summac', 'align_score', 'align',
          'vectara/hallucination_evaluation_model']

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
    # @staticmethod
    # def print(msg, c=COLOR.OKBLUE):
    #     print(f"{c}{msg}{COLOR.ENDC}")

def print_c(mgs: str, C: str):
    print(f"{C}{str}{COLOR.ENDC}")