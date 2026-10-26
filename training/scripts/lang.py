"""Language and voice settings for this Hindi fork of Paradee.

All training scripts import these instead of hard-coding Kokoro's English defaults.
Change them to train another language or voice; the architectures and losses do not depend
on the language.  Kokoro's Hindi path uses espeak-ng (`lang_code='h'`), so there is no
misaki English lexicon involved.
"""

LANG_CODE = "h"                 # Kokoro language code: h = Hindi, a = American English
VOICE = "hf_alpha"              # the voice in subhamgiri9748/kokoros-distill; hf_beta, hm_omega, hm_psi also exist
G2P_LANG = "hi"                 # espeak-ng code used by Kokoro and by the corpus checks (Kokoro maps h -> hi)
RAW_SENTS_PATH = "data/hindi_sents.json"
SENTS_PATH = "data/hindi_sents_segments.json"   # prepare_hindi.py splits the raw texts to <=400 phonemes
MAX_PHONEMES = 400              # gen_teacher.py skips longer text; Paradee's text side holds 512 tokens
STYLE_INDEX_MAX = 509           # Kokoro voice packs have 510 entries (phoneme counts 1..510)
WHISPER_LANG = "hi"             # language for the WER checks
WHISPER_MODEL = "base"          # large-v3 gives a much fairer Hindi WER if you have the memory

DEVICE = "cpu"
try:                            # corpus-prep scripts do not need torch; trainers do
    import torch
    DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    pass

# Reference sentences used by common.load_teacher to build the teacher voice pack.  They are
# not training data; any five short sentences in the target language work.
TEACHER_SENTS = [
    "भारत एक विशाल और विविधताओं से भरा देश है।",
    "आज सुबह मैंने चाय पी और अख़बार पढ़ा।",
    "विज्ञान और तकनीक ने हमारे जीवन को बहुत बदल दिया है।",
    "क्या आप कृपया मुझे रास्ता बता सकते हैं?",
    "संविधान ने सभी नागरिकों को समान अधिकार दिए हैं।",
]
