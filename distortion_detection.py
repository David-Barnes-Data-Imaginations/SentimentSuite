# Cognitive Distortion Detection Framework
# Author: David + GPT Sidekick 🧷

from typing import List, Dict, Union
import re
import json

# Define core distortions and their patterns
COGNITIVE_DISTORTIONS = [
    {
        "name": "Catastrophising",
        "patterns": [
            r"\bwhat if .* goes wrong\b",
            r"\bit will be a disaster\b",
            r"\bthis always ends badly\b",
            r"\bI can’t handle this\b",
        ],
        "description": "Imagining the worst possible outcome, no matter how unlikely."
    },
    {
        "name": "Black-and-White Thinking",
        "patterns": [
            r"\balways\b",
            r"\bnever\b",
            r"\beverything is ruined\b",
        ],
        "description": "Seeing things in absolute terms, with no middle ground."
    },
    {
        "name": "Mind Reading",
        "patterns": [
            r"\bthey must think I’m (stupid|awful|useless)\b",
            r"\bI know they hate me\b",
        ],
        "description": "Assuming you know what others are thinking without evidence."
    },
    {
        "name": "Overgeneralisation",
        "patterns": [
            r"\bI always mess things up\b",
            r"\bthis proves I’m a failure\b",
        ],
        "description": "Taking one instance and concluding it applies broadly."
    },
    {
        "name": "Emotional Reasoning",
        "patterns": [
            r"\bI feel worthless, so I must be\b",
            r"\bmy feelings are facts\b",
        ],
        "description": "Assuming that because you feel a certain way, it must be true."
    }
]

def detect_distortions(text: str) -> List[Dict[str, Union[str, int]]]:
    matches = []
    lowered = text.lower()
    for distortion in COGNITIVE_DISTORTIONS:
        for pattern in distortion["patterns"]:
            if re.search(pattern, lowered):
                matches.append({
                    "distortion": distortion["name"],
                    "description": distortion["description"],
                    "matched_pattern": pattern
                })
                break  # Only tag one match per distortion
    return matches

# Example usage:
if __name__ == "__main__":
    utterances = [
        "What if everything goes wrong tomorrow?",
        "I always mess things up.",
        "They must think I’m stupid.",
        "It’s going to be a disaster no matter what I do.",
        "I feel worthless, so I must be."
    ]
    for utt in utterances:
        print(json.dumps({"utterance": utt, "distortions": detect_distortions(utt)}, indent=2))
