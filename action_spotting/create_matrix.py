import json
import numpy as np
EVENT_DICTIONARY_V2 = {
    "Penalty": 0, 
    "Kick-off": 1, 
    "Goal": 2, 
    "Substitution": 3, 
    "Offside": 4, 
    "Shots on target": 5,
    "Shots off target": 6, 
    "Clearance": 7, 
    "Ball out of play": 8, 
    "Throw-in": 9, 
    "Foul": 10,
    "Indirect free-kick": 11, 
    "Direct free-kick": 12, 
    "Corner": 13, 
    "Yellow card": 14, 
    "Red card": 15, 
    "Yellow->red card": 16
}



with open("data/england_epl/2014-2015/2015-05-17 - 18-00 Manchester United 1 - 1 Arsenal/results_spotting.json", "r") as f:
    data = json.load(f)

amount_of_classes = len(EVENT_DICTIONARY_V2)

amount_of_frames = 0
for item in data:
    if item["frame"] > amount_of_frames:
        amount_of_frames = item["frame"]
amount_of_frames += 1

matrix = np.zeros((amount_of_frames, amount_of_classes))

for item in data:
    frame = item["frame"]
    label = item["label"]
    label_id = EVENT_DICTIONARY_V2[label]
    confidence = item["score"]

    matrix[frame, label_id] = confidence


# Print frames where events likely occurred (confidence >= 0.8)
print("\nDetected events (confidence >= 0.8):")
print("Frame\tEvent\t\t\tConfidence")
print("-" * 50)

for frame in range(amount_of_frames):
    for label_id in range(amount_of_classes):
        confidence = matrix[frame, label_id]
        if confidence >= 0.8:
            # Get event name by looking up label_id in EVENT_DICTIONARY_V2
            event_name = [k for k,v in EVENT_DICTIONARY_V2.items() if v == label_id][0]
            print(f"{frame}\t{event_name:<20}\t{confidence:.3f}")

