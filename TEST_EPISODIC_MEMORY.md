# Episodic Memory Test Procedures

## Project: GEM (Gemini Episodic Memory)

### Overview

This document provides test procedures to verify that GEM implements **human-like episodic memory** based on Tulving's (1972) framework. Each test validates one of the five dimensions of episodic memory.

---

## Tulving's Episodic Memory Framework

Human episodic memory enables us to mentally "travel back in time" to re-experience past events. It consists of five dimensions:

| Dimension | Human Example | GEM Implementation |
|-----------|---------------|-------------------|
| **WHAT** | "I saw my keys" | Object + Activity detection |
| **WHERE** | "On the kitchen counter" | Scene location + spatial position |
| **WHEN** | "This morning around 8 AM" | Timestamps + time-based queries |
| **WHO** | "I was with John" | Audio names + visual person detection |
| **HOW** | "I put them there after shopping" | Movement tracking + causal narratives |

---

## Test Environment Setup

```bash
# On Raspberry Pi Zero 2W with Whisplay HAT
export GEMINI_API_KEY=your_key_here
```

### Terminal 1: Start the Daemon (background capture)
```bash
# Captures camera + mic continuously, stores memories
python gem.py --headless
```

### Terminal 2: Run Search Mode (interactive queries)
```bash
# Uses LCD + speaker for interactive testing
python gem.py search
```

---

## Test 1: WHAT Dimension (Object Detection)

### Purpose
Verify that GEM detects and remembers physical objects in the scene.

### Test Procedure

1. **Setup**: Place several objects in view of the camera
   - Keys on a table
   - Phone on a desk
   - Glasses near a laptop
   - Medication bottle on kitchen counter

2. **Wait**: Let daemon capture at least one frame (30 seconds)

3. **Query**: In search mode, ask:
   ```
   "where are my keys?"
   "find my phone"
   "glasses"
   ```

4. **Expected Result**:
   - Object found with bounding box displayed
   - Location reported (e.g., "kitchen counter")
   - Position in frame (e.g., "center-left")
   - Confidence score shown

### Pass Criteria
- [ ] Objects detected with >70% confidence
- [ ] Correct location identified
- [ ] Bounding box displayed on LCD
- [ ] TTS speaks the result

---

## Test 2: WHAT Dimension (Activity Detection)

### Purpose
Verify that GEM detects and remembers ACTIONS, not just static objects.

### Test Procedure

1. **Setup**: Perform visible activities in front of camera:
   - Take a pill/medication (simulated)
   - Drink from a cup
   - Put on glasses
   - Place keys on counter

2. **Wait**: Let daemon capture during the activity

3. **Query**: In search mode, ask:
   ```
   "did I take my medication?"
   "did I drink my coffee?"
   "did I put on my glasses?"
   ```

4. **Expected Result**:
   - "YES!" with activity name
   - Time when activity was detected
   - Location where it happened

### Pass Criteria
- [ ] Activity detected (e.g., "taking medication")
- [ ] Correct time reported
- [ ] Location context provided
- [ ] TTS confirms the activity

---

## Test 3: WHERE Dimension (Location Memory)

### Purpose
Verify that GEM remembers WHERE objects and activities occurred.

### Test Procedure

1. **Setup**: Move objects to different locations over time:
   - Keys: front door → kitchen → desk
   - Phone: bedroom → living room

2. **Wait**: Let daemon capture at each location

3. **Query**: In search mode, ask:
   ```
   "what was on the kitchen counter?"
   "what was in the living room?"
   "where are my keys now?"
   ```

4. **Expected Result**:
   - Scene query returns objects at that location
   - Object query returns current location
   - Movement history shows location changes

### Pass Criteria
- [ ] Location correctly identified for each scene
- [ ] Multiple objects at same location listed together
- [ ] Movement between locations tracked

---

## Test 4: WHEN Dimension (Temporal Memory)

### Purpose
Verify that GEM understands and responds to time-based queries.

### Test Procedure

1. **Setup**: Let daemon run for several hours, capturing various activities

2. **Query**: In search mode, ask time-based questions:
   ```
   "what did I do this morning?"
   "what happened in the last hour?"
   "what did I see yesterday?"
   "what was I doing at 2pm?"
   ```

3. **Expected Result**:
   - List of memories from that time period
   - Activity summary narrative
   - Objects and locations from that time

### Pass Criteria
- [ ] Correct time window filtering
- [ ] Natural language time parsing works ("this morning" → 6am-12pm)
- [ ] Activity summary generated
- [ ] Memories sorted by time

---

## Test 5: WHO Dimension (People Memory)

### Purpose
Verify that GEM captures and remembers people from **both audio AND visual detection**.

### Test Procedure

1. **Setup A - Audio Names**: Have conversations in front of the device where names are mentioned:
   - "Hi John, how are you?"
   - "Sarah said she would come later"
   - "Nice to meet you, Dr. Smith"

2. **Setup B - Visual Persons**: Have people visible in front of the camera:
   - Person wearing a blue shirt
   - Person with glasses
   - Multiple people in frame

3. **Wait**: Let daemon capture audio AND video during interaction

4. **Query**: In search mode, ask:
   ```
   "who did I meet today?"
   "did I see John?"
   "did I see anyone in a blue shirt?"
   "who was I talking to this morning?"
   ```

5. **Expected Result**:
   - Names extracted from conversations (audio)
   - Visual descriptions of people seen (vision)
   - Linked identities when audio names match visual persons
   - Time and location of meeting

### Pass Criteria
- [ ] Names correctly extracted from speech (audio)
- [ ] Visual person descriptions detected (e.g., "man in blue shirt")
- [ ] Audio names linked to visual persons when both present
- [ ] Associated with correct memory/location
- [ ] General query lists all people (audio + visual)
- [ ] Specific query works for both names and descriptions

---

## Test 6: HOW Dimension (Causal Memory)

### Purpose
Verify that GEM tracks object movements and explains causation.

### Test Procedure

1. **Setup**: Move an object through multiple locations:
   - Put keys on kitchen counter
   - Move keys to jacket pocket
   - Place keys on desk

2. **Wait**: Let daemon capture at each step

3. **Query**: In search mode, ask:
   ```
   "where are my keys?"
   ```

4. **Expected Result**:
   - Current location shown
   - Movement history displayed:
     - "kitchen counter → jacket pocket (2h ago)"
     - "jacket pocket → desk (30m ago)"
   - Causal narrative: "Your keys are on the desk. You moved them from your jacket pocket 30 minutes ago."

### Pass Criteria
- [ ] Movement history recorded
- [ ] Temporal sequence correct
- [ ] Causal narrative generated
- [ ] Explains "how it got there"

---

## Test 7: Co-occurrence Memory (Contextual Binding)

### Purpose
Verify that GEM remembers objects that were together (contextual binding).

### Test Procedure

1. **Setup**: Place related objects together:
   - Keys + wallet + phone on desk
   - Glasses + book on nightstand

2. **Wait**: Let daemon capture

3. **Query**: In search mode, ask:
   ```
   "what was near my keys?"
   "what was with my glasses?"
   ```

4. **Expected Result**:
   - List of co-occurring objects
   - Location where they were together

### Pass Criteria
- [ ] Co-occurring objects listed
- [ ] Correct groupings identified
- [ ] Location context provided

---

## Test 8: Retrieval Reinforcement (Memory Strength)

### Purpose
Verify that frequently accessed memories are strengthened (Ebbinghaus curve).

### Test Procedure

1. **Setup**:
   - Create many memories (run daemon for hours)
   - Search for specific objects multiple times

2. **Check**: View memory scores:
   ```bash
   python gem.py list
   ```

3. **Expected Result**:
   - Frequently searched objects have higher access counts
   - These memories survive cleanup longer

### Pass Criteria
- [ ] Access count increases with searches
- [ ] High-access memories persist during cleanup
- [ ] Low-access old memories are forgotten

---

## Test 9: Smart Suggestions (World Knowledge)

### Purpose
Verify that GEM provides helpful suggestions when object not found.

### Test Procedure

1. **Setup**: Search for an object that hasn't been seen:
   ```
   "where are my car keys?"  (when not captured)
   ```

2. **Expected Result**:
   - "Not found" message
   - Gemini-powered suggestions:
     - "Try looking by the front door"
     - "Check your jacket pocket"
     - "Look on the kitchen counter"

### Pass Criteria
- [ ] Suggestions provided for unknown objects
- [ ] Suggestions are contextually appropriate
- [ ] Gemini world knowledge used (not hardcoded)

---

## Test 10: Real-Time Performance

### Purpose
Verify that GEM meets real-time requirements on Raspberry Pi Zero 2W.

### Test Procedure

1. **Measure**: Time each operation:
   - Voice query transcription
   - Query understanding (NLU)
   - Memory search
   - Result display + TTS

2. **Expected Result**:
   - Total query-to-answer: <10 seconds (free tier)
   - LCD displays intermediate feedback
   - User knows system is working

### Pass Criteria
- [ ] Query understood within 3 seconds
- [ ] Search completes within 2 seconds
- [ ] TTS response within 5 seconds
- [ ] Visual feedback during processing

---

## Test Summary Checklist

| Test | Dimension | Status |
|------|-----------|--------|
| 1 | WHAT (Objects) | [ ] Pass |
| 2 | WHAT (Activities) | [ ] Pass |
| 3 | WHERE (Location) | [ ] Pass |
| 4 | WHEN (Time) | [ ] Pass |
| 5 | WHO (People) | [ ] Pass |
| 6 | HOW (Causation) | [ ] Pass |
| 7 | Co-occurrence | [ ] Pass |
| 8 | Retrieval Reinforcement | [ ] Pass |
| 9 | Smart Suggestions | [ ] Pass |
| 10 | Real-Time Performance | [ ] Pass |

---

## Comparison: Human vs GEM Episodic Memory

| Feature | Human Memory | GEM Implementation |
|---------|--------------|-------------------|
| Encoding | Automatic, continuous | Camera + mic every 30s |
| Storage | Neural networks | JSON + temporal graph |
| Retrieval | Associative cues | Hash lookup + Gemini NLU |
| Forgetting | Decay over time | Cleanup based on recency + access |
| Strengthening | Repeated recall | Access count tracking |
| Context binding | Hippocampal | Co-occurrence index |
| Time awareness | Subjective | ISO timestamps + fuzzy parsing |
| Causal reasoning | Episodic replay | Movement history + narratives |
| People recognition | Face + voice | Audio names + visual descriptions |

---

## Known Limitations

1. **Visual-only activities**: Activities must be visible to camera (can't detect if you took a pill out of frame)

2. **Audio quality**: Name extraction depends on clear speech near microphone

3. **No face recognition**: Visual person detection describes appearance (clothing, features) but does not identify faces for privacy; identity requires audio names

4. **Continuous capture**: Memories only exist for captured moments (gaps between frames)

5. **No emotional valence**: Human memories are stronger when emotional; GEM treats all equally

6. **Single perspective**: Camera has fixed viewpoint; humans have mobile attention

---

## Future Enhancements

1. **MedGemma Integration**: Medical-grade memory assistance
2. **Caregiver Notifications**: Alert when user seems confused
3. **Medication Reminders**: Proactive reminders based on activity patterns
4. **Multi-camera Support**: Cover more of the environment
5. **Emotional Context**: Detect and weight emotionally significant events

