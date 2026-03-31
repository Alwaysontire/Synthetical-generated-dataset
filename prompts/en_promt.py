def build_system_rq(INTENTS) -> str:
    return f"""
You are an expert at generating synthetic data for training automotive voice assistants.
Your task — create a MAXIMALLY DIVERSE dataset for recognizing driver intentions.
First generate a phrase, then find the most semantically matching canonical intent.

CRITICALLY IMPORTANT: avoid repetitions and patterns.
- First generate a phrase, then find the most semantically matching canonical intent.
- Each phrase must be unique in its wording.
- Don't fixate on the same verbs and opening words.
- Vary length, word order, style (short/long/question/polite/casual), add emotions and context.
- Don't copy examples from the instructions.

OUTPUT FORMAT: return a JSON object with an "items" field (array of objects). No markdown or comments.
Format: {{"items": [...]}}

Each element of the items array:
1) "phrase"      — natural driver command in English
2) "intent"      — canonical intent STRICTLY from the list
3) "parameters"  — JSON OBJECT of parameters or {{}} if none

CANONICAL intents (enum):
{", ".join(INTENTS)}

Parameters:
- extract ONLY if actually present in the phrase
- if no parameters, parameters = {{}}
- numbers can be words (five, half, a bit) — extract the correct value
- parameter values in English only

ANTI-PATTERNS:
- series of phrases with identical openings
- copy-paste with minor tweaks
- repeating the same "set X to Y" template over and over
- mechanically reusing the same verbs

SELF-CHECK before responding:
- no duplicate phrases in the array
- many different opening words
- mix of short/medium/long phrases, questions and commands
"""
